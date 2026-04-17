//! Garbage collector for the Genovo scripting runtime.
//!
//! Implements a mark-and-sweep garbage collector with generational support for
//! managing script objects. The GC is designed for game engine use where
//! predictable pause times are important.
//!
//! # Architecture
//!
//! The GC uses tri-color marking (white, gray, black) with a write barrier
//! to correctly handle mutations during marking. Objects are allocated into
//! one of two generations:
//!
//! - **Nursery** (young generation): new objects land here. Collected frequently
//!   with low pause times since most objects die young.
//! - **Old generation**: objects that survive nursery collections are promoted.
//!   Collected less frequently.
//!
//! # Features
//!
//! - Mark-and-sweep with tri-color invariant
//! - Generational collection (nursery + old generation)
//! - Configurable thresholds for nursery size and promotion
//! - Write barrier for cross-generation references
//! - GC roots: stack references, global variables, native handles
//! - GC statistics: collection count, pause time, memory recovered
//! - Incremental marking support (split work across frames)
//! - Finalization callbacks for cleanup of native resources
//!
//! # Example
//!
//! ```ignore
//! let mut gc = GarbageCollector::new(GcConfig::default());
//!
//! let obj = gc.alloc(GcObjectData::String("hello".into()));
//! gc.push_root(obj);
//!
//! gc.collect_nursery();
//! println!("{}", gc.stats());
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default nursery size (number of objects before minor collection).
const DEFAULT_NURSERY_THRESHOLD: usize = 1024;

/// Default old generation size (number of objects before major collection).
const DEFAULT_OLD_GEN_THRESHOLD: usize = 8192;

/// Number of nursery collections an object must survive for promotion.
const DEFAULT_PROMOTION_AGE: u8 = 3;

/// Maximum incremental marking steps per frame.
const DEFAULT_INCREMENTAL_STEPS: usize = 256;

/// Default number of stats history entries to retain.
const DEFAULT_STATS_HISTORY: usize = 100;

/// Growth factor for dynamic threshold adjustment.
const THRESHOLD_GROWTH_FACTOR: f64 = 1.5;

/// Minimum threshold for nursery.
const MIN_NURSERY_THRESHOLD: usize = 256;

/// Minimum threshold for old generation.
const MIN_OLD_GEN_THRESHOLD: usize = 2048;

// ---------------------------------------------------------------------------
// GC handle (object identifier)
// ---------------------------------------------------------------------------

/// A handle to a GC-managed object. Handles are indices into the object store
/// combined with a generation counter for detecting stale references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GcHandle {
    /// Index into the object store.
    index: u32,
    /// Generation counter for ABA detection.
    generation: u32,
}

impl GcHandle {
    /// Create a new GC handle.
    fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Returns the raw index.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Returns the generation counter.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// A null handle (invalid).
    pub fn null() -> Self {
        Self {
            index: u32::MAX,
            generation: 0,
        }
    }

    /// Returns `true` if this is a null handle.
    pub fn is_null(&self) -> bool {
        self.index == u32::MAX
    }
}

impl fmt::Display for GcHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "GcHandle(null)")
        } else {
            write!(f, "GcHandle({}:{})", self.index, self.generation)
        }
    }
}

// ---------------------------------------------------------------------------
// Tri-color marking
// ---------------------------------------------------------------------------

/// The color of an object in the tri-color marking scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GcColor {
    /// White: not yet visited. Will be collected if still white at sweep time.
    White,
    /// Gray: discovered but children not yet scanned.
    Gray,
    /// Black: fully scanned (object and all children are reachable).
    Black,
}

impl fmt::Display for GcColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::White => write!(f, "White"),
            Self::Gray => write!(f, "Gray"),
            Self::Black => write!(f, "Black"),
        }
    }
}

// ---------------------------------------------------------------------------
// Object generation
// ---------------------------------------------------------------------------

/// Which generation an object belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Generation {
    /// Young generation (nursery).
    Nursery,
    /// Old generation.
    Old,
}

impl fmt::Display for Generation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nursery => write!(f, "Nursery"),
            Self::Old => write!(f, "Old"),
        }
    }
}

// ---------------------------------------------------------------------------
// GC Object data (what the object holds)
// ---------------------------------------------------------------------------

/// The data contained in a GC-managed object.
#[derive(Debug, Clone)]
pub enum GcObjectData {
    /// A string value.
    String(String),
    /// An array of GC handles (references to other objects) or inline values.
    Array(Vec<GcValue>),
    /// A map from string keys to GC values.
    Map(Vec<(String, GcValue)>),
    /// A script struct instance with named fields.
    Struct(StructInstance),
    /// A closure (captures references to other objects).
    Closure(ClosureData),
    /// An opaque native object with a drop callback.
    Native(NativeObject),
    /// An iterator state.
    Iterator(IteratorState),
}

/// A value that can be stored inline or as a reference to a GC object.
#[derive(Debug, Clone)]
pub enum GcValue {
    /// Nil/null.
    Nil,
    /// Boolean.
    Bool(bool),
    /// Integer.
    Int(i64),
    /// Float.
    Float(f64),
    /// A reference to a GC-managed object.
    Ref(GcHandle),
}

impl GcValue {
    /// If this value is a reference, returns the handle.
    pub fn as_ref(&self) -> Option<GcHandle> {
        match self {
            GcValue::Ref(h) => Some(*h),
            _ => None,
        }
    }

    /// Returns `true` if this value contains a GC reference.
    pub fn is_ref(&self) -> bool {
        matches!(self, GcValue::Ref(_))
    }
}

/// A script struct instance.
#[derive(Debug, Clone)]
pub struct StructInstance {
    /// The struct type name.
    pub type_name: String,
    /// Fields stored as (name, value) pairs.
    pub fields: Vec<(String, GcValue)>,
}

impl StructInstance {
    /// Create a new struct instance.
    pub fn new(type_name: &str) -> Self {
        Self {
            type_name: type_name.to_string(),
            fields: Vec::new(),
        }
    }

    /// Set a field value.
    pub fn set_field(&mut self, name: &str, value: GcValue) {
        for (n, v) in &mut self.fields {
            if n == name {
                *v = value;
                return;
            }
        }
        self.fields.push((name.to_string(), value));
    }

    /// Get a field value.
    pub fn get_field(&self, name: &str) -> Option<&GcValue> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}

/// Data for a script closure.
#[derive(Debug, Clone)]
pub struct ClosureData {
    /// The function identifier.
    pub function_id: u32,
    /// Captured values (upvalues).
    pub captures: Vec<GcValue>,
}

/// An opaque native object managed by the GC.
#[derive(Debug, Clone)]
pub struct NativeObject {
    /// Type identifier string.
    pub type_id: String,
    /// Opaque data (serialized or represented as bytes).
    pub data: Vec<u8>,
    /// Whether this object needs finalization.
    pub needs_finalization: bool,
}

/// An iterator state.
#[derive(Debug, Clone)]
pub struct IteratorState {
    /// The collection being iterated.
    pub source: GcHandle,
    /// Current index.
    pub position: usize,
    /// Whether the iterator is exhausted.
    pub exhausted: bool,
}

// ---------------------------------------------------------------------------
// GC Object (internal representation)
// ---------------------------------------------------------------------------

/// Internal GC object with metadata.
struct GcObject {
    /// The actual data.
    data: Option<GcObjectData>,
    /// Current marking color.
    color: GcColor,
    /// Which generation this object belongs to.
    generation: Generation,
    /// Number of nursery collections survived.
    age: u8,
    /// Generation counter for the slot (for handle validation).
    slot_generation: u32,
    /// Whether this object is a GC root.
    is_root: bool,
    /// Whether this object has been finalized.
    finalized: bool,
    /// Approximate size in bytes (for memory tracking).
    size_bytes: usize,
}

impl GcObject {
    fn new(data: GcObjectData, generation: Generation, slot_generation: u32) -> Self {
        let size_bytes = estimate_size(&data);
        Self {
            data: Some(data),
            color: GcColor::White,
            generation,
            age: 0,
            slot_generation,
            is_root: false,
            finalized: false,
            size_bytes,
        }
    }

    fn is_alive(&self) -> bool {
        self.data.is_some()
    }
}

/// Estimate the memory size of an object's data.
fn estimate_size(data: &GcObjectData) -> usize {
    match data {
        GcObjectData::String(s) => 24 + s.len(),
        GcObjectData::Array(arr) => 24 + arr.len() * 16,
        GcObjectData::Map(entries) => {
            24 + entries.iter().map(|(k, _)| 16 + k.len()).sum::<usize>()
        }
        GcObjectData::Struct(s) => {
            24 + s.type_name.len()
                + s.fields.iter().map(|(n, _)| 16 + n.len()).sum::<usize>()
        }
        GcObjectData::Closure(c) => 24 + c.captures.len() * 16,
        GcObjectData::Native(n) => 24 + n.type_id.len() + n.data.len(),
        GcObjectData::Iterator(_) => 32,
    }
}

/// Collect GC references from object data.
fn collect_references(data: &GcObjectData) -> Vec<GcHandle> {
    let mut refs = Vec::new();
    match data {
        GcObjectData::String(_) => {}
        GcObjectData::Array(arr) => {
            for val in arr {
                if let GcValue::Ref(h) = val {
                    refs.push(*h);
                }
            }
        }
        GcObjectData::Map(entries) => {
            for (_, val) in entries {
                if let GcValue::Ref(h) = val {
                    refs.push(*h);
                }
            }
        }
        GcObjectData::Struct(s) => {
            for (_, val) in &s.fields {
                if let GcValue::Ref(h) = val {
                    refs.push(*h);
                }
            }
        }
        GcObjectData::Closure(c) => {
            for val in &c.captures {
                if let GcValue::Ref(h) = val {
                    refs.push(*h);
                }
            }
        }
        GcObjectData::Native(_) => {}
        GcObjectData::Iterator(it) => {
            if !it.source.is_null() {
                refs.push(it.source);
            }
        }
    }
    refs
}

// ---------------------------------------------------------------------------
// GC Configuration
// ---------------------------------------------------------------------------

/// Configuration for the garbage collector.
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Number of nursery objects before triggering a minor collection.
    pub nursery_threshold: usize,
    /// Number of old-gen objects before triggering a major collection.
    pub old_gen_threshold: usize,
    /// Number of nursery collections an object must survive for promotion.
    pub promotion_age: u8,
    /// Maximum incremental marking steps per call.
    pub incremental_steps: usize,
    /// Whether to enable generational collection.
    pub generational: bool,
    /// Whether to dynamically adjust thresholds based on allocation rate.
    pub adaptive_thresholds: bool,
    /// Whether to collect statistics.
    pub collect_stats: bool,
    /// How many stats snapshots to retain.
    pub stats_history_size: usize,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            nursery_threshold: DEFAULT_NURSERY_THRESHOLD,
            old_gen_threshold: DEFAULT_OLD_GEN_THRESHOLD,
            promotion_age: DEFAULT_PROMOTION_AGE,
            incremental_steps: DEFAULT_INCREMENTAL_STEPS,
            generational: true,
            adaptive_thresholds: true,
            collect_stats: true,
            stats_history_size: DEFAULT_STATS_HISTORY,
        }
    }
}

impl GcConfig {
    /// Create a config optimized for low-latency (smaller nursery, more frequent collections).
    pub fn low_latency() -> Self {
        Self {
            nursery_threshold: 512,
            old_gen_threshold: 4096,
            promotion_age: 2,
            incremental_steps: 128,
            ..Self::default()
        }
    }

    /// Create a config optimized for throughput (larger nursery, less frequent collections).
    pub fn high_throughput() -> Self {
        Self {
            nursery_threshold: 4096,
            old_gen_threshold: 32768,
            promotion_age: 5,
            incremental_steps: 512,
            ..Self::default()
        }
    }

    /// Create a simple non-generational config.
    pub fn simple() -> Self {
        Self {
            generational: false,
            nursery_threshold: 2048,
            old_gen_threshold: 2048,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// GC Statistics
// ---------------------------------------------------------------------------

/// Statistics from a single GC collection cycle.
#[derive(Debug, Clone)]
pub struct GcCollectionStats {
    /// Type of collection (minor or major).
    pub collection_type: CollectionType,
    /// How long the collection took.
    pub pause_duration: Duration,
    /// Number of objects scanned (marked).
    pub objects_scanned: usize,
    /// Number of objects collected (freed).
    pub objects_collected: usize,
    /// Bytes recovered.
    pub bytes_recovered: usize,
    /// Number of objects promoted to old generation.
    pub objects_promoted: usize,
    /// Total objects alive after collection.
    pub objects_alive: usize,
    /// Total memory used after collection.
    pub memory_used: usize,
}

/// The type of a GC collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollectionType {
    /// Minor collection (nursery only).
    Minor,
    /// Major collection (all generations).
    Major,
}

impl fmt::Display for CollectionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minor => write!(f, "Minor"),
            Self::Major => write!(f, "Major"),
        }
    }
}

/// Aggregate GC statistics.
#[derive(Debug, Clone)]
pub struct GcStats {
    /// Total number of minor collections.
    pub minor_collections: u64,
    /// Total number of major collections.
    pub major_collections: u64,
    /// Total time spent in GC pauses.
    pub total_pause_time: Duration,
    /// Longest single GC pause.
    pub max_pause_time: Duration,
    /// Average GC pause time.
    pub avg_pause_time: Duration,
    /// Total objects allocated.
    pub total_allocated: u64,
    /// Total objects collected.
    pub total_collected: u64,
    /// Total bytes recovered.
    pub total_bytes_recovered: u64,
    /// Total objects promoted.
    pub total_promoted: u64,
    /// Current nursery size.
    pub nursery_size: usize,
    /// Current old generation size.
    pub old_gen_size: usize,
    /// Current total memory used.
    pub memory_used: usize,
    /// Number of roots.
    pub root_count: usize,
    /// Write barrier triggers.
    pub write_barrier_triggers: u64,
    /// History of recent collection stats.
    pub history: Vec<GcCollectionStats>,
}

impl GcStats {
    fn new() -> Self {
        Self {
            minor_collections: 0,
            major_collections: 0,
            total_pause_time: Duration::ZERO,
            max_pause_time: Duration::ZERO,
            avg_pause_time: Duration::ZERO,
            total_allocated: 0,
            total_collected: 0,
            total_bytes_recovered: 0,
            total_promoted: 0,
            nursery_size: 0,
            old_gen_size: 0,
            memory_used: 0,
            root_count: 0,
            write_barrier_triggers: 0,
            history: Vec::new(),
        }
    }

    fn record_collection(&mut self, stats: GcCollectionStats, history_max: usize) {
        match stats.collection_type {
            CollectionType::Minor => self.minor_collections += 1,
            CollectionType::Major => self.major_collections += 1,
        }

        self.total_pause_time += stats.pause_duration;
        if stats.pause_duration > self.max_pause_time {
            self.max_pause_time = stats.pause_duration;
        }
        self.total_collected += stats.objects_collected as u64;
        self.total_bytes_recovered += stats.bytes_recovered as u64;
        self.total_promoted += stats.objects_promoted as u64;

        let total_collections = self.minor_collections + self.major_collections;
        if total_collections > 0 {
            self.avg_pause_time = self.total_pause_time / total_collections as u32;
        }

        if self.history.len() >= history_max {
            self.history.remove(0);
        }
        self.history.push(stats);
    }

    /// Returns the collection rate (collections per second of total pause time).
    pub fn collection_rate(&self) -> f64 {
        let total = self.minor_collections + self.major_collections;
        if self.total_pause_time.as_secs_f64() > 0.0 {
            total as f64 / self.total_pause_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Returns the survival rate (fraction of objects that survive collection).
    pub fn survival_rate(&self) -> f64 {
        if self.total_allocated == 0 {
            return 0.0;
        }
        let survived = self.total_allocated - self.total_collected;
        survived as f64 / self.total_allocated as f64
    }
}

impl fmt::Display for GcStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GC[minor={}, major={}, alloc={}, collected={}, mem={:.1}KB, \
             max_pause={:.2}ms, avg_pause={:.2}ms]",
            self.minor_collections,
            self.major_collections,
            self.total_allocated,
            self.total_collected,
            self.memory_used as f64 / 1024.0,
            self.max_pause_time.as_secs_f64() * 1000.0,
            self.avg_pause_time.as_secs_f64() * 1000.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Garbage Collector
// ---------------------------------------------------------------------------

/// The garbage collector.
pub struct GarbageCollector {
    /// Object store.
    objects: Vec<GcObject>,
    /// Free list (indices of slots available for reuse).
    free_list: Vec<u32>,
    /// GC roots (handles that should not be collected).
    roots: HashSet<GcHandle>,
    /// Stack roots (temporary roots from the VM stack).
    stack_roots: Vec<GcHandle>,
    /// Global variable roots.
    global_roots: HashMap<String, GcHandle>,
    /// Gray set for incremental marking.
    gray_set: VecDeque<u32>,
    /// Remembered set: old objects that reference nursery objects.
    remembered_set: HashSet<u32>,
    /// Configuration.
    config: GcConfig,
    /// Statistics.
    stats: GcStats,
    /// Current nursery allocation count.
    nursery_count: usize,
    /// Current old gen allocation count.
    old_gen_count: usize,
    /// Total memory used (approximate).
    memory_used: usize,
    /// Finalization callbacks.
    finalizers: HashMap<String, Box<dyn Fn(&NativeObject)>>,
    /// Whether a collection is currently in progress (for incremental GC).
    marking_in_progress: bool,
}

impl GarbageCollector {
    /// Create a new garbage collector with the given configuration.
    pub fn new(config: GcConfig) -> Self {
        Self {
            objects: Vec::with_capacity(config.nursery_threshold),
            free_list: Vec::new(),
            roots: HashSet::new(),
            stack_roots: Vec::new(),
            global_roots: HashMap::new(),
            gray_set: VecDeque::new(),
            remembered_set: HashSet::new(),
            config,
            stats: GcStats::new(),
            nursery_count: 0,
            old_gen_count: 0,
            memory_used: 0,
            finalizers: HashMap::new(),
            marking_in_progress: false,
        }
    }

    /// Create a GC with default configuration.
    pub fn default_gc() -> Self {
        Self::new(GcConfig::default())
    }

    // --- Allocation ---

    /// Allocate a new GC object. Returns a handle. May trigger collection.
    pub fn alloc(&mut self, data: GcObjectData) -> GcHandle {
        // Check if we need to collect.
        if self.nursery_count >= self.config.nursery_threshold {
            self.collect_nursery();
        }

        self.alloc_no_collect(data, Generation::Nursery)
    }

    /// Allocate without triggering collection (internal use).
    fn alloc_no_collect(&mut self, data: GcObjectData, generation: Generation) -> GcHandle {
        let size = estimate_size(&data);

        let (index, slot_gen) = if let Some(free_idx) = self.free_list.pop() {
            let slot_gen = self.objects[free_idx as usize].slot_generation + 1;
            (free_idx, slot_gen)
        } else {
            let idx = self.objects.len() as u32;
            self.objects.push(GcObject {
                data: None,
                color: GcColor::White,
                generation: Generation::Nursery,
                age: 0,
                slot_generation: 0,
                is_root: false,
                finalized: false,
                size_bytes: 0,
            });
            (idx, 0u32)
        };

        let obj = GcObject::new(data, generation, slot_gen);
        self.objects[index as usize] = obj;

        match generation {
            Generation::Nursery => self.nursery_count += 1,
            Generation::Old => self.old_gen_count += 1,
        }
        self.memory_used += size;
        self.stats.total_allocated += 1;

        GcHandle::new(index, slot_gen)
    }

    // --- Root management ---

    /// Add a GC handle as a root (prevents collection).
    pub fn push_root(&mut self, handle: GcHandle) {
        self.roots.insert(handle);
    }

    /// Remove a GC handle from roots.
    pub fn pop_root(&mut self, handle: &GcHandle) {
        self.roots.remove(handle);
    }

    /// Set a stack root (used by the VM during execution).
    pub fn set_stack_roots(&mut self, roots: Vec<GcHandle>) {
        self.stack_roots = roots;
    }

    /// Clear all stack roots.
    pub fn clear_stack_roots(&mut self) {
        self.stack_roots.clear();
    }

    /// Set a named global root.
    pub fn set_global_root(&mut self, name: &str, handle: GcHandle) {
        self.global_roots.insert(name.to_string(), handle);
    }

    /// Remove a named global root.
    pub fn remove_global_root(&mut self, name: &str) {
        self.global_roots.remove(name);
    }

    // --- Object access ---

    /// Get an immutable reference to an object's data.
    pub fn get(&self, handle: GcHandle) -> Option<&GcObjectData> {
        if handle.is_null() {
            return None;
        }
        let idx = handle.index as usize;
        if idx >= self.objects.len() {
            return None;
        }
        let obj = &self.objects[idx];
        if obj.slot_generation != handle.generation {
            return None; // Stale handle.
        }
        obj.data.as_ref()
    }

    /// Get a mutable reference to an object's data.
    pub fn get_mut(&mut self, handle: GcHandle) -> Option<&mut GcObjectData> {
        if handle.is_null() {
            return None;
        }
        let idx = handle.index as usize;
        if idx >= self.objects.len() {
            return None;
        }
        let obj = &mut self.objects[idx];
        if obj.slot_generation != handle.generation {
            return None;
        }
        obj.data.as_mut()
    }

    /// Returns `true` if a handle refers to a valid, alive object.
    pub fn is_alive(&self, handle: GcHandle) -> bool {
        self.get(handle).is_some()
    }

    // --- Write barrier ---

    /// Write barrier: must be called when an old-gen object gains a reference
    /// to a nursery object. This adds the source to the remembered set.
    pub fn write_barrier(&mut self, source: GcHandle, _target: GcHandle) {
        if source.is_null() {
            return;
        }
        let idx = source.index as usize;
        if idx < self.objects.len() {
            let obj = &self.objects[idx];
            if obj.generation == Generation::Old {
                self.remembered_set.insert(source.index);
                self.stats.write_barrier_triggers += 1;
            }
        }
    }

    // --- Collection ---

    /// Perform a minor (nursery) collection.
    pub fn collect_nursery(&mut self) {
        if !self.config.generational {
            self.collect_full();
            return;
        }

        let start = Instant::now();

        // Mark phase: start from roots and remembered set.
        self.mark_roots_for_nursery();
        self.mark_propagate();

        // Sweep nursery.
        let (collected, bytes_recovered, promoted) = self.sweep_nursery();

        let pause = start.elapsed();

        let stats = GcCollectionStats {
            collection_type: CollectionType::Minor,
            pause_duration: pause,
            objects_scanned: self.count_marked(),
            objects_collected: collected,
            bytes_recovered,
            objects_promoted: promoted,
            objects_alive: self.nursery_count + self.old_gen_count,
            memory_used: self.memory_used,
        };

        self.update_stats(stats);
        self.reset_colors();
        self.remembered_set.clear();

        // Check if we should do a major collection.
        if self.old_gen_count >= self.config.old_gen_threshold {
            self.collect_full();
        }

        // Adaptive threshold.
        if self.config.adaptive_thresholds {
            self.adjust_thresholds();
        }
    }

    /// Perform a full (major) collection of all generations.
    pub fn collect_full(&mut self) {
        let start = Instant::now();

        // Mark all roots.
        self.mark_all_roots();
        self.mark_propagate();

        // Sweep all objects.
        let (collected, bytes_recovered) = self.sweep_all();

        let pause = start.elapsed();

        let stats = GcCollectionStats {
            collection_type: CollectionType::Major,
            pause_duration: pause,
            objects_scanned: self.count_marked(),
            objects_collected: collected,
            bytes_recovered,
            objects_promoted: 0,
            objects_alive: self.nursery_count + self.old_gen_count,
            memory_used: self.memory_used,
        };

        self.update_stats(stats);
        self.reset_colors();
        self.remembered_set.clear();
    }

    /// Mark roots for a nursery collection (only scan nursery objects).
    fn mark_roots_for_nursery(&mut self) {
        // Mark from explicit roots.
        let root_indices: Vec<u32> = self.roots.iter().map(|r| r.index).collect();
        for idx in root_indices {
            self.mark_gray(idx);
        }
        // Mark from stack roots.
        let stack_indices: Vec<u32> = self.stack_roots.iter().map(|r| r.index).collect();
        for idx in stack_indices {
            self.mark_gray(idx);
        }
        // Mark from global roots.
        let globals: Vec<u32> = self.global_roots.values().map(|h| h.index).collect();
        for idx in globals {
            self.mark_gray(idx);
        }
        // Mark from remembered set (old objects referencing nursery).
        let remembered: Vec<u32> = self.remembered_set.iter().copied().collect();
        for idx in remembered {
            self.mark_gray(idx);
        }
    }

    /// Mark all roots (for full collection).
    fn mark_all_roots(&mut self) {
        let root_indices: Vec<u32> = self.roots.iter().map(|r| r.index).collect();
        for idx in root_indices {
            self.mark_gray(idx);
        }
        let stack_indices: Vec<u32> = self.stack_roots.iter().map(|r| r.index).collect();
        for idx in stack_indices {
            self.mark_gray(idx);
        }
        let globals: Vec<u32> = self.global_roots.values().map(|h| h.index).collect();
        for idx in globals {
            self.mark_gray(idx);
        }
    }

    /// Mark an object as gray (discovered).
    fn mark_gray(&mut self, index: u32) {
        let idx = index as usize;
        if idx >= self.objects.len() {
            return;
        }
        if !self.objects[idx].is_alive() {
            return;
        }
        if self.objects[idx].color != GcColor::White {
            return;
        }
        self.objects[idx].color = GcColor::Gray;
        self.gray_set.push_back(index);
    }

    /// Propagate marks through the gray set until empty.
    fn mark_propagate(&mut self) {
        while let Some(index) = self.gray_set.pop_front() {
            let idx = index as usize;
            if idx >= self.objects.len() || !self.objects[idx].is_alive() {
                continue;
            }
            if self.objects[idx].color != GcColor::Gray {
                continue;
            }

            // Collect references from this object.
            let refs = if let Some(ref data) = self.objects[idx].data {
                collect_references(data)
            } else {
                Vec::new()
            };

            // Mark this object as black (fully scanned).
            self.objects[idx].color = GcColor::Black;

            // Mark all referenced objects as gray.
            for child in refs {
                self.mark_gray(child.index);
            }
        }
    }

    /// Incremental marking: process up to N gray objects per call.
    pub fn mark_incremental(&mut self, max_steps: usize) -> bool {
        let steps = max_steps.min(self.gray_set.len());
        for _ in 0..steps {
            if let Some(index) = self.gray_set.pop_front() {
                let idx = index as usize;
                if idx >= self.objects.len() || !self.objects[idx].is_alive() {
                    continue;
                }
                if self.objects[idx].color != GcColor::Gray {
                    continue;
                }
                let refs = if let Some(ref data) = self.objects[idx].data {
                    collect_references(data)
                } else {
                    Vec::new()
                };
                self.objects[idx].color = GcColor::Black;
                for child in refs {
                    self.mark_gray(child.index);
                }
            }
        }
        self.gray_set.is_empty()
    }

    /// Sweep nursery objects: collect white objects, promote survivors.
    fn sweep_nursery(&mut self) -> (usize, usize, usize) {
        let mut collected = 0;
        let mut bytes_recovered = 0;
        let mut promoted = 0;

        for i in 0..self.objects.len() {
            if !self.objects[i].is_alive() {
                continue;
            }
            if self.objects[i].generation != Generation::Nursery {
                continue;
            }

            if self.objects[i].color == GcColor::White {
                // Object is unreachable — collect it.
                self.finalize_object(i);
                bytes_recovered += self.objects[i].size_bytes;
                self.memory_used -= self.objects[i].size_bytes;
                self.objects[i].data = None;
                self.free_list.push(i as u32);
                self.nursery_count -= 1;
                collected += 1;
            } else {
                // Object survived — increment age and maybe promote.
                self.objects[i].age += 1;
                if self.objects[i].age >= self.config.promotion_age {
                    self.objects[i].generation = Generation::Old;
                    self.nursery_count -= 1;
                    self.old_gen_count += 1;
                    promoted += 1;
                }
            }
        }

        (collected, bytes_recovered, promoted)
    }

    /// Sweep all objects: collect all white objects.
    fn sweep_all(&mut self) -> (usize, usize) {
        let mut collected = 0;
        let mut bytes_recovered = 0;

        for i in 0..self.objects.len() {
            if !self.objects[i].is_alive() {
                continue;
            }

            if self.objects[i].color == GcColor::White {
                self.finalize_object(i);
                bytes_recovered += self.objects[i].size_bytes;
                self.memory_used -= self.objects[i].size_bytes;
                match self.objects[i].generation {
                    Generation::Nursery => self.nursery_count -= 1,
                    Generation::Old => self.old_gen_count -= 1,
                }
                self.objects[i].data = None;
                self.free_list.push(i as u32);
                collected += 1;
            }
        }

        (collected, bytes_recovered)
    }

    /// Finalize an object (call the finalization callback for native objects).
    fn finalize_object(&mut self, index: usize) {
        if self.objects[index].finalized {
            return;
        }
        self.objects[index].finalized = true;

        // Check for native finalizer.
        if let Some(GcObjectData::Native(ref native)) = self.objects[index].data {
            if native.needs_finalization {
                if let Some(finalizer) = self.finalizers.get(&native.type_id) {
                    finalizer(native);
                }
            }
        }
    }

    /// Reset all object colors to white (prepare for next collection).
    fn reset_colors(&mut self) {
        for obj in &mut self.objects {
            if obj.is_alive() {
                obj.color = GcColor::White;
            }
        }
    }

    /// Count how many objects are marked (non-white).
    fn count_marked(&self) -> usize {
        self.objects
            .iter()
            .filter(|o| o.is_alive() && o.color != GcColor::White)
            .count()
    }

    /// Update statistics after a collection.
    fn update_stats(&mut self, collection_stats: GcCollectionStats) {
        if self.config.collect_stats {
            self.stats.nursery_size = self.nursery_count;
            self.stats.old_gen_size = self.old_gen_count;
            self.stats.memory_used = self.memory_used;
            self.stats.root_count =
                self.roots.len() + self.stack_roots.len() + self.global_roots.len();
            self.stats
                .record_collection(collection_stats, self.config.stats_history_size);
        }
    }

    /// Dynamically adjust thresholds based on allocation patterns.
    fn adjust_thresholds(&mut self) {
        // If the nursery fills up quickly, increase the threshold.
        if self.nursery_count as f64 > self.config.nursery_threshold as f64 * 0.9 {
            self.config.nursery_threshold =
                ((self.config.nursery_threshold as f64 * THRESHOLD_GROWTH_FACTOR) as usize)
                    .max(MIN_NURSERY_THRESHOLD);
        }
        if self.old_gen_count as f64 > self.config.old_gen_threshold as f64 * 0.9 {
            self.config.old_gen_threshold =
                ((self.config.old_gen_threshold as f64 * THRESHOLD_GROWTH_FACTOR) as usize)
                    .max(MIN_OLD_GEN_THRESHOLD);
        }
    }

    // --- Finalizer registration ---

    /// Register a finalization callback for native objects of a given type.
    pub fn register_finalizer<F>(&mut self, type_id: &str, callback: F)
    where
        F: Fn(&NativeObject) + 'static,
    {
        self.finalizers
            .insert(type_id.to_string(), Box::new(callback));
    }

    // --- Statistics ---

    /// Returns the current GC statistics.
    pub fn stats(&self) -> &GcStats {
        &self.stats
    }

    /// Returns the number of live objects.
    pub fn live_object_count(&self) -> usize {
        self.nursery_count + self.old_gen_count
    }

    /// Returns the nursery object count.
    pub fn nursery_count(&self) -> usize {
        self.nursery_count
    }

    /// Returns the old generation object count.
    pub fn old_gen_count(&self) -> usize {
        self.old_gen_count
    }

    /// Returns the approximate memory used by managed objects.
    pub fn memory_used(&self) -> usize {
        self.memory_used
    }

    /// Returns the number of roots.
    pub fn root_count(&self) -> usize {
        self.roots.len() + self.stack_roots.len() + self.global_roots.len()
    }

    /// Force a full collection.
    pub fn force_collect(&mut self) {
        self.collect_full();
    }
}

impl fmt::Display for GarbageCollector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GC[nursery={}, old={}, mem={:.1}KB, roots={}]",
            self.nursery_count,
            self.old_gen_count,
            self.memory_used as f64 / 1024.0,
            self.root_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gc_handle() {
        let h = GcHandle::new(0, 0);
        assert!(!h.is_null());
        assert_eq!(h.index(), 0);

        let null = GcHandle::null();
        assert!(null.is_null());
    }

    #[test]
    fn test_gc_alloc_and_get() {
        let mut gc = GarbageCollector::new(GcConfig::default());
        let h = gc.alloc(GcObjectData::String("hello".to_string()));
        assert!(gc.is_alive(h));

        match gc.get(h) {
            Some(GcObjectData::String(s)) => assert_eq!(s, "hello"),
            _ => panic!("expected string"),
        }
    }

    #[test]
    fn test_gc_roots_prevent_collection() {
        let mut gc = GarbageCollector::new(GcConfig::simple());
        let h = gc.alloc(GcObjectData::String("keep me".to_string()));
        gc.push_root(h);
        gc.collect_full();
        assert!(gc.is_alive(h));
    }

    #[test]
    fn test_gc_unreachable_collected() {
        let mut gc = GarbageCollector::new(GcConfig::simple());
        let h = gc.alloc(GcObjectData::String("temp".to_string()));
        // No root — should be collected.
        gc.collect_full();
        assert!(!gc.is_alive(h));
    }

    #[test]
    fn test_gc_reachable_through_reference() {
        let mut gc = GarbageCollector::new(GcConfig::simple());

        let child = gc.alloc(GcObjectData::String("child".to_string()));
        let parent = gc.alloc(GcObjectData::Array(vec![GcValue::Ref(child)]));

        gc.push_root(parent);
        gc.collect_full();

        assert!(gc.is_alive(parent));
        assert!(gc.is_alive(child));
    }

    #[test]
    fn test_gc_cycle_collection() {
        let mut gc = GarbageCollector::new(GcConfig::simple());

        // Create two objects that reference each other (cycle).
        let a = gc.alloc(GcObjectData::Array(Vec::new()));
        let b = gc.alloc(GcObjectData::Array(vec![GcValue::Ref(a)]));

        // Make a reference b.
        if let Some(GcObjectData::Array(ref mut arr)) = gc.get_mut(a) {
            arr.push(GcValue::Ref(b));
        }

        // No roots — both should be collected despite the cycle.
        gc.collect_full();
        assert!(!gc.is_alive(a));
        assert!(!gc.is_alive(b));
    }

    #[test]
    fn test_gc_generational_promotion() {
        let config = GcConfig {
            nursery_threshold: 10,
            promotion_age: 2,
            generational: true,
            ..GcConfig::default()
        };
        let mut gc = GarbageCollector::new(config);

        let h = gc.alloc(GcObjectData::String("promote me".to_string()));
        gc.push_root(h);

        // Trigger nursery collections.
        gc.collect_nursery();
        gc.collect_nursery();
        gc.collect_nursery(); // Should promote after 2+ survivals.

        assert!(gc.is_alive(h));
        assert!(gc.stats().total_promoted > 0);
    }

    #[test]
    fn test_gc_struct_instance() {
        let mut gc = GarbageCollector::new(GcConfig::simple());

        let mut inst = StructInstance::new("Player");
        inst.set_field("name", GcValue::Int(42));
        inst.set_field("hp", GcValue::Float(100.0));

        let h = gc.alloc(GcObjectData::Struct(inst));
        gc.push_root(h);

        if let Some(GcObjectData::Struct(s)) = gc.get(h) {
            assert_eq!(s.type_name, "Player");
            assert!(s.get_field("name").is_some());
        } else {
            panic!("expected struct");
        }
    }

    #[test]
    fn test_gc_stats() {
        let mut gc = GarbageCollector::new(GcConfig::simple());

        for i in 0..10 {
            gc.alloc(GcObjectData::String(format!("obj_{i}")));
        }
        gc.collect_full();

        assert_eq!(gc.stats().total_allocated, 10);
        assert_eq!(gc.stats().total_collected, 10);
        assert_eq!(gc.stats().major_collections, 1);
    }

    #[test]
    fn test_gc_global_roots() {
        let mut gc = GarbageCollector::new(GcConfig::simple());

        let h = gc.alloc(GcObjectData::String("global".to_string()));
        gc.set_global_root("my_var", h);
        gc.collect_full();
        assert!(gc.is_alive(h));

        gc.remove_global_root("my_var");
        gc.collect_full();
        assert!(!gc.is_alive(h));
    }

    #[test]
    fn test_gc_handle_reuse() {
        let mut gc = GarbageCollector::new(GcConfig::simple());

        let h1 = gc.alloc(GcObjectData::String("first".to_string()));
        gc.collect_full(); // Collects h1 (no root).
        assert!(!gc.is_alive(h1));

        // Allocate again — should reuse the slot.
        let h2 = gc.alloc(GcObjectData::String("second".to_string()));
        gc.push_root(h2);
        assert!(gc.is_alive(h2));
        // h1 should still be invalid (different generation).
        assert!(!gc.is_alive(h1));
    }

    #[test]
    fn test_gc_config_variants() {
        let low_lat = GcConfig::low_latency();
        assert_eq!(low_lat.nursery_threshold, 512);

        let high_tp = GcConfig::high_throughput();
        assert_eq!(high_tp.nursery_threshold, 4096);

        let simple = GcConfig::simple();
        assert!(!simple.generational);
    }

    #[test]
    fn test_gc_value() {
        let nil = GcValue::Nil;
        assert!(!nil.is_ref());

        let handle = GcHandle::new(5, 0);
        let r = GcValue::Ref(handle);
        assert!(r.is_ref());
        assert_eq!(r.as_ref(), Some(handle));
    }

    #[test]
    fn test_gc_memory_tracking() {
        let mut gc = GarbageCollector::new(GcConfig::simple());
        assert_eq!(gc.memory_used(), 0);

        let h = gc.alloc(GcObjectData::String("hello".to_string()));
        assert!(gc.memory_used() > 0);
        let used = gc.memory_used();

        gc.collect_full(); // h not rooted, should be collected.
        assert!(gc.memory_used() < used);
    }

    #[test]
    fn test_incremental_marking() {
        let mut gc = GarbageCollector::new(GcConfig::simple());

        let root = gc.alloc(GcObjectData::Array(Vec::new()));
        gc.push_root(root);

        for _ in 0..5 {
            gc.alloc(GcObjectData::String("temp".to_string()));
        }

        gc.mark_all_roots();
        let done = gc.mark_incremental(2);
        // Should complete eventually.
        if !done {
            let done2 = gc.mark_incremental(100);
            assert!(done2);
        }
    }
}
