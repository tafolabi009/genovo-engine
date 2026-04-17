// engine/physics/src/broadphase_sap.rs
//
// Sweep-and-Prune (SAP) broad-phase collision detection for the Genovo engine.
//
// Implements an incremental sort-based broad-phase that maintains sorted lists
// of AABB endpoints along each axis. When objects move, only their endpoints
// shift in the sorted lists, and new/lost overlaps are detected by tracking
// swaps during the incremental insertion sort.
//
// Features:
// - Three-axis endpoint lists (X, Y, Z)
// - Incremental sort with swap-based pair tracking
// - Pair management with add/remove callbacks
// - Support for static and dynamic objects
// - Object sleeping (skip update for sleeping objects)
// - Pair filtering via collision layers
// - Batch insert/remove operations
// - Statistics tracking (pairs, updates, swaps)
//
// The SAP algorithm is particularly efficient for scenes where most objects
// move slowly or are static, as the sorted lists remain nearly sorted and
// the incremental sort is O(n + s) where s is the number of swaps.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of objects the SAP can handle.
pub const MAX_SAP_OBJECTS: usize = 65536;

/// Default collision layer (all bits set).
pub const DEFAULT_COLLISION_MASK: u32 = 0xFFFF_FFFF;

/// Epsilon for AABB expansion (skin width).
pub const DEFAULT_AABB_SKIN: f32 = 0.05;

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box for broad-phase collision detection.
#[derive(Debug, Clone, Copy)]
pub struct SapAabb {
    /// Minimum corner.
    pub min: [f32; 3],
    /// Maximum corner.
    pub max: [f32; 3],
}

impl SapAabb {
    /// Creates a new AABB.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Creates an AABB from center and half-extents.
    pub fn from_center_half_extents(center: [f32; 3], half_extents: [f32; 3]) -> Self {
        Self {
            min: [
                center[0] - half_extents[0],
                center[1] - half_extents[1],
                center[2] - half_extents[2],
            ],
            max: [
                center[0] + half_extents[0],
                center[1] + half_extents[1],
                center[2] + half_extents[2],
            ],
        }
    }

    /// Creates an invalid (empty) AABB.
    pub fn invalid() -> Self {
        Self {
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        }
    }

    /// Tests if two AABBs overlap on all three axes.
    pub fn overlaps(&self, other: &SapAabb) -> bool {
        self.min[0] <= other.max[0] && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1] && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2] && self.max[2] >= other.min[2]
    }

    /// Tests if this AABB contains a point.
    pub fn contains_point(&self, point: [f32; 3]) -> bool {
        point[0] >= self.min[0] && point[0] <= self.max[0]
            && point[1] >= self.min[1] && point[1] <= self.max[1]
            && point[2] >= self.min[2] && point[2] <= self.max[2]
    }

    /// Returns the center of the AABB.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Returns the half-extents of the AABB.
    pub fn half_extents(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }

    /// Returns the surface area of the AABB (for SAH heuristic).
    pub fn surface_area(&self) -> f32 {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        2.0 * (dx * dy + dy * dz + dz * dx)
    }

    /// Returns the volume of the AABB.
    pub fn volume(&self) -> f32 {
        let dx = (self.max[0] - self.min[0]).max(0.0);
        let dy = (self.max[1] - self.min[1]).max(0.0);
        let dz = (self.max[2] - self.min[2]).max(0.0);
        dx * dy * dz
    }

    /// Expands the AABB by a uniform skin width.
    pub fn expand(&self, skin: f32) -> SapAabb {
        SapAabb {
            min: [self.min[0] - skin, self.min[1] - skin, self.min[2] - skin],
            max: [self.max[0] + skin, self.max[1] + skin, self.max[2] + skin],
        }
    }

    /// Merge two AABBs into the smallest enclosing AABB.
    pub fn merge(&self, other: &SapAabb) -> SapAabb {
        SapAabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Collision pair
// ---------------------------------------------------------------------------

/// An ordered pair of colliding object IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollisionPair {
    /// Lower ID.
    pub a: u32,
    /// Higher ID.
    pub b: u32,
}

impl CollisionPair {
    /// Creates a new ordered pair (ensures a < b).
    pub fn new(id_a: u32, id_b: u32) -> Self {
        if id_a < id_b {
            Self { a: id_a, b: id_b }
        } else {
            Self { a: id_b, b: id_a }
        }
    }
}

// ---------------------------------------------------------------------------
// SAP endpoint
// ---------------------------------------------------------------------------

/// An endpoint in the sorted list (min or max of an AABB on one axis).
#[derive(Debug, Clone, Copy)]
struct Endpoint {
    /// The coordinate value.
    value: f32,
    /// The object ID this endpoint belongs to.
    object_id: u32,
    /// Whether this is a min (true) or max (false) endpoint.
    is_min: bool,
}

// ---------------------------------------------------------------------------
// SAP object
// ---------------------------------------------------------------------------

/// Data stored per object in the SAP.
#[derive(Debug, Clone)]
struct SapObject {
    /// Object identifier.
    id: u32,
    /// Current AABB.
    aabb: SapAabb,
    /// Expanded AABB (with skin).
    expanded_aabb: SapAabb,
    /// Collision layer (bitmask).
    layer: u32,
    /// Collision mask (which layers to collide with).
    mask: u32,
    /// Whether this object is sleeping.
    sleeping: bool,
    /// Whether this object is static (never moves).
    is_static: bool,
    /// Index into each axis endpoint list (min endpoint index).
    endpoint_indices: [usize; 3],
    /// User data attached to this object.
    user_data: u64,
}

// ---------------------------------------------------------------------------
// Pair callback
// ---------------------------------------------------------------------------

/// Callback type for pair events.
pub type PairCallback = Box<dyn Fn(u32, u32) + Send + Sync>;

/// Pair event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairEvent {
    /// A new overlapping pair was detected.
    Added,
    /// An overlapping pair was lost.
    Removed,
}

/// Recorded pair event for deferred processing.
#[derive(Debug, Clone)]
pub struct PairEventRecord {
    /// The pair involved.
    pub pair: CollisionPair,
    /// The event type.
    pub event: PairEvent,
}

// ---------------------------------------------------------------------------
// SAP statistics
// ---------------------------------------------------------------------------

/// Statistics for the SAP broad-phase.
#[derive(Debug, Clone, Default)]
pub struct SapStats {
    /// Total objects tracked.
    pub total_objects: usize,
    /// Active (non-sleeping) objects.
    pub active_objects: usize,
    /// Static objects.
    pub static_objects: usize,
    /// Total overlapping pairs.
    pub total_pairs: usize,
    /// Pairs added this frame.
    pub pairs_added: usize,
    /// Pairs removed this frame.
    pub pairs_removed: usize,
    /// Total swap operations during sort.
    pub total_swaps: usize,
    /// Update time in microseconds.
    pub update_time_us: u64,
    /// Number of endpoint list entries (per axis).
    pub endpoints_per_axis: usize,
}

impl SapStats {
    /// Returns a formatted summary.
    pub fn summary(&self) -> String {
        format!(
            "SAP: {} objects ({} active, {} static), {} pairs (+{} -{}, {} swaps)",
            self.total_objects,
            self.active_objects,
            self.static_objects,
            self.total_pairs,
            self.pairs_added,
            self.pairs_removed,
            self.total_swaps,
        )
    }
}

// ---------------------------------------------------------------------------
// Sweep-and-Prune broad-phase
// ---------------------------------------------------------------------------

/// Sweep-and-Prune broad-phase collision detection.
///
/// Maintains sorted endpoint lists along each of the three axes. When objects
/// move, their endpoints shift in the lists. Overlapping pairs are detected
/// by tracking which intervals overlap on all three axes simultaneously.
///
/// # Usage
///
/// ```ignore
/// let mut sap = SweepAndPrune::new(SapConfig::default());
///
/// // Add objects.
/// sap.add_object(0, aabb0, 1, 0xFFFFFFFF);
/// sap.add_object(1, aabb1, 1, 0xFFFFFFFF);
///
/// // Each frame:
/// sap.update_object(0, new_aabb0);
/// sap.update();
///
/// // Get pairs.
/// for pair in sap.pairs() {
///     // pair.a and pair.b are overlapping.
/// }
/// ```
#[derive(Debug)]
pub struct SweepAndPrune {
    /// Configuration.
    pub config: SapConfig,
    /// Sorted endpoint lists for each axis (X=0, Y=1, Z=2).
    axes: [Vec<Endpoint>; 3],
    /// Map from object ID to SapObject.
    objects: HashMap<u32, SapObject>,
    /// Current set of overlapping pairs.
    pairs: HashSet<CollisionPair>,
    /// Pair events generated during the last update.
    events: Vec<PairEventRecord>,
    /// Overlap counters per axis: (pair) -> axis overlap count.
    /// A pair is active when all 3 axes overlap (count == 3).
    axis_overlaps: HashMap<CollisionPair, u8>,
    /// Statistics.
    pub stats: SapStats,
    /// Next object ID for auto-assignment.
    next_id: u32,
    /// Whether the SAP needs a full rebuild (after many insertions).
    needs_rebuild: bool,
    /// Dirty flags for objects that moved.
    dirty_objects: HashSet<u32>,
}

/// SAP configuration.
#[derive(Debug, Clone)]
pub struct SapConfig {
    /// AABB skin width for hysteresis.
    pub aabb_skin: f32,
    /// Primary sort axis (0=X, 1=Y, 2=Z).
    pub primary_axis: usize,
    /// Whether to use multi-axis overlap confirmation.
    pub multi_axis: bool,
    /// Maximum pairs before warning.
    pub max_pairs_warning: usize,
    /// Whether to batch pair events for deferred processing.
    pub deferred_events: bool,
}

impl Default for SapConfig {
    fn default() -> Self {
        Self {
            aabb_skin: DEFAULT_AABB_SKIN,
            primary_axis: 0,
            multi_axis: true,
            max_pairs_warning: 50000,
            deferred_events: true,
        }
    }
}

impl SweepAndPrune {
    /// Creates a new SAP broad-phase.
    pub fn new(config: SapConfig) -> Self {
        Self {
            config,
            axes: [Vec::new(), Vec::new(), Vec::new()],
            objects: HashMap::new(),
            pairs: HashSet::new(),
            events: Vec::new(),
            axis_overlaps: HashMap::new(),
            stats: SapStats::default(),
            next_id: 0,
            needs_rebuild: false,
            dirty_objects: HashSet::new(),
        }
    }

    /// Returns the next available object ID.
    pub fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Adds a new object to the broad-phase.
    pub fn add_object(&mut self, id: u32, aabb: SapAabb, layer: u32, mask: u32) {
        self.add_object_with_data(id, aabb, layer, mask, false, 0);
    }

    /// Adds a new object with user data.
    pub fn add_object_with_data(
        &mut self,
        id: u32,
        aabb: SapAabb,
        layer: u32,
        mask: u32,
        is_static: bool,
        user_data: u64,
    ) {
        let expanded = aabb.expand(self.config.aabb_skin);

        // Insert endpoints for each axis.
        let mut endpoint_indices = [0usize; 3];
        for axis in 0..3 {
            let min_ep = Endpoint {
                value: expanded.min[axis],
                object_id: id,
                is_min: true,
            };
            let max_ep = Endpoint {
                value: expanded.max[axis],
                object_id: id,
                is_min: false,
            };
            endpoint_indices[axis] = self.axes[axis].len();
            self.axes[axis].push(min_ep);
            self.axes[axis].push(max_ep);
        }

        self.objects.insert(id, SapObject {
            id,
            aabb,
            expanded_aabb: expanded,
            layer,
            mask,
            sleeping: false,
            is_static,
            endpoint_indices,
            user_data,
        });

        self.dirty_objects.insert(id);
        self.needs_rebuild = true;
    }

    /// Removes an object from the broad-phase.
    pub fn remove_object(&mut self, id: u32) {
        if self.objects.remove(&id).is_none() {
            return;
        }

        // Remove endpoints from each axis.
        for axis in 0..3 {
            self.axes[axis].retain(|ep| ep.object_id != id);
        }

        // Remove all pairs involving this object.
        let pairs_to_remove: Vec<CollisionPair> = self.pairs.iter()
            .filter(|p| p.a == id || p.b == id)
            .copied()
            .collect();

        for pair in &pairs_to_remove {
            self.pairs.remove(pair);
            self.axis_overlaps.remove(pair);
            self.events.push(PairEventRecord {
                pair: *pair,
                event: PairEvent::Removed,
            });
        }

        self.dirty_objects.remove(&id);
    }

    /// Updates the AABB of an object.
    pub fn update_object(&mut self, id: u32, new_aabb: SapAabb) {
        if let Some(obj) = self.objects.get_mut(&id) {
            // Only update if the AABB moved outside the expanded AABB.
            let expanded = &obj.expanded_aabb;
            let needs_update = new_aabb.min[0] < expanded.min[0]
                || new_aabb.min[1] < expanded.min[1]
                || new_aabb.min[2] < expanded.min[2]
                || new_aabb.max[0] > expanded.max[0]
                || new_aabb.max[1] > expanded.max[1]
                || new_aabb.max[2] > expanded.max[2];

            if needs_update {
                obj.aabb = new_aabb;
                obj.expanded_aabb = new_aabb.expand(self.config.aabb_skin);
                self.dirty_objects.insert(id);
            }
        }
    }

    /// Set an object as sleeping (skip updates until woken).
    pub fn set_sleeping(&mut self, id: u32, sleeping: bool) {
        if let Some(obj) = self.objects.get_mut(&id) {
            obj.sleeping = sleeping;
            if !sleeping {
                self.dirty_objects.insert(id);
            }
        }
    }

    /// Perform the broad-phase update: sort endpoints and detect pair changes.
    pub fn update(&mut self) {
        self.events.clear();
        self.stats = SapStats::default();
        self.stats.total_objects = self.objects.len();
        self.stats.active_objects = self.objects.values().filter(|o| !o.sleeping).count();
        self.stats.static_objects = self.objects.values().filter(|o| o.is_static).count();

        if self.needs_rebuild {
            self.full_rebuild();
            self.needs_rebuild = false;
        } else {
            self.incremental_update();
        }

        self.stats.total_pairs = self.pairs.len();
        self.stats.endpoints_per_axis = self.axes[0].len();
    }

    /// Full rebuild: re-sort all endpoint lists and recompute all pairs.
    fn full_rebuild(&mut self) {
        // Update all endpoint values from objects.
        for axis in 0..3 {
            for ep in &mut self.axes[axis] {
                if let Some(obj) = self.objects.get(&ep.object_id) {
                    if ep.is_min {
                        ep.value = obj.expanded_aabb.min[axis];
                    } else {
                        ep.value = obj.expanded_aabb.max[axis];
                    }
                }
            }

            // Full sort.
            self.axes[axis].sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Rebuild all pairs using sweep on the primary axis.
        let old_pairs = std::mem::take(&mut self.pairs);
        self.axis_overlaps.clear();

        self.sweep_and_collect_pairs();

        // Generate events for new and removed pairs.
        for pair in &self.pairs {
            if !old_pairs.contains(pair) {
                self.events.push(PairEventRecord {
                    pair: *pair,
                    event: PairEvent::Added,
                });
            }
        }
        for pair in &old_pairs {
            if !self.pairs.contains(pair) {
                self.events.push(PairEventRecord {
                    pair: *pair,
                    event: PairEvent::Removed,
                });
            }
        }

        self.stats.pairs_added = self.events.iter().filter(|e| e.event == PairEvent::Added).count();
        self.stats.pairs_removed = self.events.iter().filter(|e| e.event == PairEvent::Removed).count();
    }

    /// Sweep the primary axis and collect overlapping pairs, verified on all axes.
    fn sweep_and_collect_pairs(&mut self) {
        let axis = self.config.primary_axis;
        let endpoints = &self.axes[axis];

        // Active set: objects whose interval is currently open.
        let mut active: Vec<u32> = Vec::new();

        for ep in endpoints {
            if ep.is_min {
                // This object's interval starts: check against all active objects.
                for &active_id in &active {
                    let pair = CollisionPair::new(ep.object_id, active_id);
                    if self.should_collide(pair.a, pair.b) {
                        // Verify overlap on all axes.
                        if self.verify_all_axes_overlap(pair.a, pair.b) {
                            self.pairs.insert(pair);
                        }
                    }
                }
                active.push(ep.object_id);
            } else {
                // This object's interval ends: remove from active set.
                if let Some(pos) = active.iter().position(|&id| id == ep.object_id) {
                    active.swap_remove(pos);
                }
            }
        }
    }

    /// Check if two objects should collide based on layer/mask filtering.
    fn should_collide(&self, id_a: u32, id_b: u32) -> bool {
        if id_a == id_b {
            return false;
        }

        let obj_a = match self.objects.get(&id_a) {
            Some(o) => o,
            None => return false,
        };
        let obj_b = match self.objects.get(&id_b) {
            Some(o) => o,
            None => return false,
        };

        // Skip static-static pairs.
        if obj_a.is_static && obj_b.is_static {
            return false;
        }

        // Layer/mask filtering.
        (obj_a.layer & obj_b.mask) != 0 && (obj_b.layer & obj_a.mask) != 0
    }

    /// Verify that two objects overlap on all three axes.
    fn verify_all_axes_overlap(&self, id_a: u32, id_b: u32) -> bool {
        let obj_a = match self.objects.get(&id_a) {
            Some(o) => o,
            None => return false,
        };
        let obj_b = match self.objects.get(&id_b) {
            Some(o) => o,
            None => return false,
        };

        obj_a.expanded_aabb.overlaps(&obj_b.expanded_aabb)
    }

    /// Incremental update: only update dirty objects and re-sort locally.
    fn incremental_update(&mut self) {
        let dirty_ids: Vec<u32> = self.dirty_objects.drain().collect();

        if dirty_ids.is_empty() {
            return;
        }

        // Update endpoint values for dirty objects.
        for axis in 0..3 {
            for ep in &mut self.axes[axis] {
                if dirty_ids.contains(&ep.object_id) {
                    if let Some(obj) = self.objects.get(&ep.object_id) {
                        if ep.is_min {
                            ep.value = obj.expanded_aabb.min[axis];
                        } else {
                            ep.value = obj.expanded_aabb.max[axis];
                        }
                    }
                }
            }
        }

        // Incremental insertion sort on each axis, tracking swaps.
        let mut swaps = 0usize;
        for axis in 0..3 {
            swaps += self.insertion_sort_axis(axis);
        }
        self.stats.total_swaps = swaps;

        // Revalidate all existing pairs.
        let current_pairs: Vec<CollisionPair> = self.pairs.iter().copied().collect();
        for pair in &current_pairs {
            if !self.verify_all_axes_overlap(pair.a, pair.b) || !self.should_collide(pair.a, pair.b) {
                self.pairs.remove(pair);
                self.events.push(PairEventRecord {
                    pair: *pair,
                    event: PairEvent::Removed,
                });
            }
        }

        // Check for new pairs involving dirty objects.
        for &dirty_id in &dirty_ids {
            if let Some(obj) = self.objects.get(&dirty_id) {
                if obj.sleeping {
                    continue;
                }
                for (&other_id, other_obj) in &self.objects {
                    if other_id == dirty_id {
                        continue;
                    }
                    let pair = CollisionPair::new(dirty_id, other_id);
                    if !self.pairs.contains(&pair) && self.should_collide(pair.a, pair.b) {
                        if obj.expanded_aabb.overlaps(&other_obj.expanded_aabb) {
                            self.pairs.insert(pair);
                            self.events.push(PairEventRecord {
                                pair,
                                event: PairEvent::Added,
                            });
                        }
                    }
                }
            }
        }

        self.stats.pairs_added = self.events.iter().filter(|e| e.event == PairEvent::Added).count();
        self.stats.pairs_removed = self.events.iter().filter(|e| e.event == PairEvent::Removed).count();
    }

    /// Insertion sort on a single axis, returning the number of swaps.
    fn insertion_sort_axis(&mut self, axis: usize) -> usize {
        let list = &mut self.axes[axis];
        let mut swaps = 0;

        for i in 1..list.len() {
            let mut j = i;
            while j > 0 && list[j].value < list[j - 1].value {
                list.swap(j, j - 1);
                swaps += 1;
                j -= 1;
            }
        }

        swaps
    }

    /// Returns the current set of overlapping pairs.
    pub fn pairs(&self) -> &HashSet<CollisionPair> {
        &self.pairs
    }

    /// Returns the pair events from the last update.
    pub fn events(&self) -> &[PairEventRecord] {
        &self.events
    }

    /// Returns only the added pairs from the last update.
    pub fn added_pairs(&self) -> Vec<CollisionPair> {
        self.events.iter()
            .filter(|e| e.event == PairEvent::Added)
            .map(|e| e.pair)
            .collect()
    }

    /// Returns only the removed pairs from the last update.
    pub fn removed_pairs(&self) -> Vec<CollisionPair> {
        self.events.iter()
            .filter(|e| e.event == PairEvent::Removed)
            .map(|e| e.pair)
            .collect()
    }

    /// Returns the number of objects.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Returns the number of overlapping pairs.
    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// Returns the AABB of an object.
    pub fn get_aabb(&self, id: u32) -> Option<SapAabb> {
        self.objects.get(&id).map(|o| o.aabb)
    }

    /// Returns the user data of an object.
    pub fn get_user_data(&self, id: u32) -> Option<u64> {
        self.objects.get(&id).map(|o| o.user_data)
    }

    /// Sets the user data of an object.
    pub fn set_user_data(&mut self, id: u32, data: u64) {
        if let Some(obj) = self.objects.get_mut(&id) {
            obj.user_data = data;
        }
    }

    /// Sets the collision layer of an object.
    pub fn set_layer(&mut self, id: u32, layer: u32) {
        if let Some(obj) = self.objects.get_mut(&id) {
            obj.layer = layer;
            self.dirty_objects.insert(id);
        }
    }

    /// Sets the collision mask of an object.
    pub fn set_mask(&mut self, id: u32, mask: u32) {
        if let Some(obj) = self.objects.get_mut(&id) {
            obj.mask = mask;
            self.dirty_objects.insert(id);
        }
    }

    /// Query all objects overlapping a given AABB.
    pub fn query_aabb(&self, query: &SapAabb) -> Vec<u32> {
        let mut results = Vec::new();
        for (id, obj) in &self.objects {
            if obj.expanded_aabb.overlaps(query) {
                results.push(*id);
            }
        }
        results
    }

    /// Query all objects overlapping a given AABB with layer filtering.
    pub fn query_aabb_filtered(&self, query: &SapAabb, layer_mask: u32) -> Vec<u32> {
        let mut results = Vec::new();
        for (id, obj) in &self.objects {
            if (obj.layer & layer_mask) != 0 && obj.expanded_aabb.overlaps(query) {
                results.push(*id);
            }
        }
        results
    }

    /// Ray query: find objects whose AABB intersects a ray.
    pub fn query_ray(&self, origin: [f32; 3], direction: [f32; 3], max_dist: f32) -> Vec<(u32, f32)> {
        let mut results = Vec::new();
        let inv_dir = [
            if direction[0].abs() > 1e-10 { 1.0 / direction[0] } else { f32::MAX },
            if direction[1].abs() > 1e-10 { 1.0 / direction[1] } else { f32::MAX },
            if direction[2].abs() > 1e-10 { 1.0 / direction[2] } else { f32::MAX },
        ];

        for (id, obj) in &self.objects {
            if let Some(t) = ray_aabb_intersect(&origin, &inv_dir, &obj.expanded_aabb) {
                if t >= 0.0 && t <= max_dist {
                    results.push((*id, t));
                }
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get statistics.
    pub fn statistics(&self) -> &SapStats {
        &self.stats
    }

    /// Reset the broad-phase, removing all objects and pairs.
    pub fn clear(&mut self) {
        self.axes = [Vec::new(), Vec::new(), Vec::new()];
        self.objects.clear();
        self.pairs.clear();
        self.events.clear();
        self.axis_overlaps.clear();
        self.dirty_objects.clear();
        self.stats = SapStats::default();
    }
}

// ---------------------------------------------------------------------------
// Ray-AABB intersection helper
// ---------------------------------------------------------------------------

/// Ray-AABB intersection using the slab method.
fn ray_aabb_intersect(origin: &[f32; 3], inv_dir: &[f32; 3], aabb: &SapAabb) -> Option<f32> {
    let mut tmin = f32::NEG_INFINITY;
    let mut tmax = f32::INFINITY;

    for axis in 0..3 {
        let t1 = (aabb.min[axis] - origin[axis]) * inv_dir[axis];
        let t2 = (aabb.max[axis] - origin[axis]) * inv_dir[axis];

        let t_near = t1.min(t2);
        let t_far = t1.max(t2);

        tmin = tmin.max(t_near);
        tmax = tmax.min(t_far);
    }

    if tmin <= tmax && tmax >= 0.0 {
        Some(tmin.max(0.0))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Multi-SAP (for large worlds with spatial partitioning)
// ---------------------------------------------------------------------------

/// A multi-SAP that divides the world into grid cells, each containing
/// a local SAP for efficient broad-phase in very large worlds.
#[derive(Debug)]
pub struct MultiSap {
    /// Cell size.
    pub cell_size: f32,
    /// Grid of SAP instances indexed by cell coordinate.
    cells: HashMap<(i32, i32, i32), SweepAndPrune>,
    /// Map from object ID to the cells it occupies.
    object_cells: HashMap<u32, Vec<(i32, i32, i32)>>,
    /// Global pair set (aggregated from all cells).
    global_pairs: HashSet<CollisionPair>,
    /// Configuration for child SAPs.
    sap_config: SapConfig,
}

impl MultiSap {
    /// Creates a new multi-SAP with the given cell size.
    pub fn new(cell_size: f32, config: SapConfig) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
            object_cells: HashMap::new(),
            global_pairs: HashSet::new(),
            sap_config: config,
        }
    }

    /// Convert a world position to a cell coordinate.
    fn world_to_cell(&self, pos: f32, axis: usize) -> i32 {
        (pos / self.cell_size).floor() as i32
    }

    /// Get the cells covered by an AABB.
    fn aabb_cells(&self, aabb: &SapAabb) -> Vec<(i32, i32, i32)> {
        let min_cell = (
            self.world_to_cell(aabb.min[0], 0),
            self.world_to_cell(aabb.min[1], 1),
            self.world_to_cell(aabb.min[2], 2),
        );
        let max_cell = (
            self.world_to_cell(aabb.max[0], 0),
            self.world_to_cell(aabb.max[1], 1),
            self.world_to_cell(aabb.max[2], 2),
        );

        let mut cells = Vec::new();
        for x in min_cell.0..=max_cell.0 {
            for y in min_cell.1..=max_cell.1 {
                for z in min_cell.2..=max_cell.2 {
                    cells.push((x, y, z));
                }
            }
        }
        cells
    }

    /// Add an object to the multi-SAP.
    pub fn add_object(&mut self, id: u32, aabb: SapAabb, layer: u32, mask: u32) {
        let cells = self.aabb_cells(&aabb);
        for &cell in &cells {
            let sap = self.cells.entry(cell).or_insert_with(|| {
                SweepAndPrune::new(self.sap_config.clone())
            });
            sap.add_object(id, aabb, layer, mask);
        }
        self.object_cells.insert(id, cells);
    }

    /// Remove an object from the multi-SAP.
    pub fn remove_object(&mut self, id: u32) {
        if let Some(cells) = self.object_cells.remove(&id) {
            for cell in cells {
                if let Some(sap) = self.cells.get_mut(&cell) {
                    sap.remove_object(id);
                }
            }
        }
    }

    /// Update an object's AABB.
    pub fn update_object(&mut self, id: u32, new_aabb: SapAabb) {
        // Re-insert into the correct cells.
        self.remove_object(id);
        // We need layer/mask from the old object. Default to full collision.
        self.add_object(id, new_aabb, DEFAULT_COLLISION_MASK, DEFAULT_COLLISION_MASK);
    }

    /// Perform the broad-phase update.
    pub fn update(&mut self) {
        self.global_pairs.clear();
        for sap in self.cells.values_mut() {
            sap.update();
            self.global_pairs.extend(sap.pairs().iter());
        }
    }

    /// Returns the global set of overlapping pairs.
    pub fn pairs(&self) -> &HashSet<CollisionPair> {
        &self.global_pairs
    }

    /// Returns the number of active cells.
    pub fn active_cell_count(&self) -> usize {
        self.cells.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_overlap() {
        let a = SapAabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = SapAabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        assert!(a.overlaps(&b));

        let c = SapAabb::new([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_aabb_center_and_extents() {
        let aabb = SapAabb::new([1.0, 2.0, 3.0], [5.0, 6.0, 7.0]);
        let center = aabb.center();
        assert!((center[0] - 3.0).abs() < 1e-5);
        assert!((center[1] - 4.0).abs() < 1e-5);
        assert!((center[2] - 5.0).abs() < 1e-5);

        let he = aabb.half_extents();
        assert!((he[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_collision_pair_ordering() {
        let p1 = CollisionPair::new(5, 3);
        let p2 = CollisionPair::new(3, 5);
        assert_eq!(p1, p2);
        assert_eq!(p1.a, 3);
        assert_eq!(p1.b, 5);
    }

    #[test]
    fn test_sap_add_and_detect() {
        let mut sap = SweepAndPrune::new(SapConfig::default());
        sap.add_object(0, SapAabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), 1, 0xFFFFFFFF);
        sap.add_object(1, SapAabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]), 1, 0xFFFFFFFF);
        sap.update();

        assert_eq!(sap.pair_count(), 1);
        assert!(sap.pairs().contains(&CollisionPair::new(0, 1)));
    }

    #[test]
    fn test_sap_no_overlap() {
        let mut sap = SweepAndPrune::new(SapConfig::default());
        sap.add_object(0, SapAabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 1, 0xFFFFFFFF);
        sap.add_object(1, SapAabb::new([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]), 1, 0xFFFFFFFF);
        sap.update();

        assert_eq!(sap.pair_count(), 0);
    }

    #[test]
    fn test_sap_remove_object() {
        let mut sap = SweepAndPrune::new(SapConfig::default());
        sap.add_object(0, SapAabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), 1, 0xFFFFFFFF);
        sap.add_object(1, SapAabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]), 1, 0xFFFFFFFF);
        sap.update();
        assert_eq!(sap.pair_count(), 1);

        sap.remove_object(1);
        sap.update();
        assert_eq!(sap.pair_count(), 0);
    }

    #[test]
    fn test_sap_layer_filtering() {
        let mut sap = SweepAndPrune::new(SapConfig::default());
        sap.add_object(0, SapAabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), 1, 2); // Layer 1, only collides with layer 2.
        sap.add_object(1, SapAabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]), 4, 0xFFFFFFFF); // Layer 4.
        sap.update();

        // Layer 1 & mask 0xFFFFFFFF = OK for B, but layer 4 & mask 2 = 0 for A -> no collision.
        assert_eq!(sap.pair_count(), 0);
    }

    #[test]
    fn test_sap_static_static_skip() {
        let mut sap = SweepAndPrune::new(SapConfig::default());
        sap.add_object_with_data(0, SapAabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), 1, 0xFFFFFFFF, true, 0);
        sap.add_object_with_data(1, SapAabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]), 1, 0xFFFFFFFF, true, 0);
        sap.update();

        // Static-static pairs are skipped.
        assert_eq!(sap.pair_count(), 0);
    }

    #[test]
    fn test_sap_update_and_separate() {
        let mut sap = SweepAndPrune::new(SapConfig { aabb_skin: 0.0, ..Default::default() });
        sap.add_object(0, SapAabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), 1, 0xFFFFFFFF);
        sap.add_object(1, SapAabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]), 1, 0xFFFFFFFF);
        sap.update();
        assert_eq!(sap.pair_count(), 1);

        // Move object 1 far away.
        sap.update_object(1, SapAabb::new([10.0, 10.0, 10.0], [12.0, 12.0, 12.0]));
        sap.update();
        assert_eq!(sap.pair_count(), 0);
    }

    #[test]
    fn test_ray_aabb_intersect() {
        let aabb = SapAabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        let origin = [0.0, 2.0, 2.0];
        let direction = [1.0, 0.0, 0.0];
        let inv_dir = [1.0 / direction[0], f32::MAX, f32::MAX];
        let t = ray_aabb_intersect(&origin, &inv_dir, &aabb);
        assert!(t.is_some());
        assert!((t.unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sap_query_aabb() {
        let mut sap = SweepAndPrune::new(SapConfig::default());
        sap.add_object(0, SapAabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 1, 0xFFFFFFFF);
        sap.add_object(1, SapAabb::new([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]), 1, 0xFFFFFFFF);
        sap.update();

        let query = SapAabb::new([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]);
        let results = sap.query_aabb(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0);
    }

    #[test]
    fn test_sap_stats() {
        let mut sap = SweepAndPrune::new(SapConfig::default());
        sap.add_object(0, SapAabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), 1, 0xFFFFFFFF);
        sap.add_object(1, SapAabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]), 1, 0xFFFFFFFF);
        sap.update();

        assert_eq!(sap.stats.total_objects, 2);
        assert_eq!(sap.stats.total_pairs, 1);
    }
}
