//! World-scale navigation system for the Genovo engine.
//!
//! Provides a complete navigation subsystem for AI pathfinding across
//! large, streaming worlds:
//!
//! - **Global navmesh management** — register, update, and query navmeshes.
//! - **Navmesh streaming** — load/unload navmesh tiles per world cell.
//! - **Cross-cell pathfinding** — find paths that span multiple navmesh tiles.
//! - **Dynamic obstacle registration** — add/remove obstacles at runtime.
//! - **Navigation query caching** — LRU cache for frequent path queries.
//! - **Path corridor following** — smooth path following with re-planning.
//! - **Off-mesh links** — ladders, teleporters, jumps, and other special
//!   connections between navmesh polygons.

use std::collections::{HashMap, HashSet, BinaryHeap, VecDeque};
use std::fmt;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

/// Minimal 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    pub fn length(&self) -> f32 { (self.x * self.x + self.y * self.y + self.z * self.z).sqrt() }

    pub fn distance(&self, o: &Self) -> f32 {
        let dx = self.x - o.x; let dy = self.y - o.y; let dz = self.z - o.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn distance_xz(&self, o: &Self) -> f32 {
        let dx = self.x - o.x; let dz = self.z - o.z;
        (dx * dx + dz * dz).sqrt()
    }

    pub fn add(&self, o: &Self) -> Self { Self::new(self.x + o.x, self.y + o.y, self.z + o.z) }
    pub fn sub(&self, o: &Self) -> Self { Self::new(self.x - o.x, self.y - o.y, self.z - o.z) }
    pub fn scale(&self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }
    pub fn lerp(&self, o: &Self, t: f32) -> Self {
        Self::new(
            self.x + (o.x - self.x) * t,
            self.y + (o.y - self.y) * t,
            self.z + (o.z - self.z) * t,
        )
    }
    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-9 { Self::ZERO } else { Self::new(self.x / len, self.y / len, self.z / len) }
    }
    pub fn dot(&self, o: &Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }
}

impl Default for Vec3 { fn default() -> Self { Self::ZERO } }

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Unique ID for a navmesh polygon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolyId(u64);

impl PolyId {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, AtomicOrdering::Relaxed))
    }
    pub fn from_raw(raw: u64) -> Self { Self(raw) }
    pub fn raw(&self) -> u64 { self.0 }
}

/// Unique ID for an off-mesh link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinkId(u64);

impl LinkId {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, AtomicOrdering::Relaxed))
    }
    pub fn from_raw(raw: u64) -> Self { Self(raw) }
    pub fn raw(&self) -> u64 { self.0 }
}

/// Unique ID for a dynamic obstacle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObstacleId(u64);

impl ObstacleId {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, AtomicOrdering::Relaxed))
    }
    pub fn from_raw(raw: u64) -> Self { Self(raw) }
}

/// World cell coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellCoord {
    pub x: i32,
    pub z: i32,
}

impl CellCoord {
    pub fn new(x: i32, z: i32) -> Self { Self { x, z } }

    pub fn chebyshev_distance(&self, other: &Self) -> u32 {
        (self.x - other.x).unsigned_abs().max((self.z - other.z).unsigned_abs())
    }
}

impl fmt::Display for CellCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.x, self.z)
    }
}

// ---------------------------------------------------------------------------
// Navmesh polygon
// ---------------------------------------------------------------------------

/// A single polygon in the navigation mesh.
#[derive(Debug, Clone)]
pub struct NavPoly {
    /// Unique polygon id.
    pub id: PolyId,
    /// Vertices of the polygon (convex, CCW).
    pub vertices: Vec<Vec3>,
    /// Neighbor polygon ids (one per edge, None if boundary).
    pub neighbors: Vec<Option<PolyId>>,
    /// Center of the polygon.
    pub center: Vec3,
    /// Area of the polygon.
    pub area: f32,
    /// Navigation cost multiplier (1.0 = normal).
    pub cost: f32,
    /// Flags (walkable, swimable, etc.).
    pub flags: NavFlags,
    /// Cell this polygon belongs to.
    pub cell: CellCoord,
}

impl NavPoly {
    /// Creates a new navigation polygon.
    pub fn new(id: PolyId, vertices: Vec<Vec3>, cell: CellCoord) -> Self {
        let center = Self::compute_center(&vertices);
        let area = Self::compute_area(&vertices);
        let neighbor_count = vertices.len();

        Self {
            id,
            vertices,
            neighbors: vec![None; neighbor_count],
            center,
            area,
            cost: 1.0,
            flags: NavFlags::WALKABLE,
            cell,
        }
    }

    fn compute_center(vertices: &[Vec3]) -> Vec3 {
        let n = vertices.len() as f32;
        let sum_x: f32 = vertices.iter().map(|v| v.x).sum();
        let sum_y: f32 = vertices.iter().map(|v| v.y).sum();
        let sum_z: f32 = vertices.iter().map(|v| v.z).sum();
        Vec3::new(sum_x / n, sum_y / n, sum_z / n)
    }

    fn compute_area(vertices: &[Vec3]) -> f32 {
        if vertices.len() < 3 { return 0.0; }
        let mut area = 0.0f32;
        for i in 1..vertices.len() - 1 {
            let a = vertices[i].sub(&vertices[0]);
            let b = vertices[i + 1].sub(&vertices[0]);
            let cross_y = a.x * b.z - a.z * b.x;
            area += cross_y.abs();
        }
        area * 0.5
    }

    /// Whether a point (XZ plane) is inside this polygon.
    pub fn contains_point_xz(&self, p: Vec3) -> bool {
        let n = self.vertices.len();
        if n < 3 { return false; }

        for i in 0..n {
            let j = (i + 1) % n;
            let edge = Vec3::new(
                self.vertices[j].x - self.vertices[i].x,
                0.0,
                self.vertices[j].z - self.vertices[i].z,
            );
            let to_point = Vec3::new(
                p.x - self.vertices[i].x,
                0.0,
                p.z - self.vertices[i].z,
            );
            let cross = edge.x * to_point.z - edge.z * to_point.x;
            if cross < 0.0 { return false; }
        }

        true
    }

    /// Returns the closest point on the polygon boundary to a given point.
    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        if self.contains_point_xz(p) {
            // Project onto polygon plane (approximate: use center Y).
            return Vec3::new(p.x, self.center.y, p.z);
        }

        // Find closest point on edges.
        let mut best = self.center;
        let mut best_dist = f32::MAX;

        let n = self.vertices.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let closest = closest_point_on_segment(p, self.vertices[i], self.vertices[j]);
            let dist = p.distance_xz(&closest);
            if dist < best_dist {
                best_dist = dist;
                best = closest;
            }
        }

        best
    }
}

/// Closest point on a line segment to a given point.
fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b.sub(&a);
    let ap = p.sub(&a);
    let t = (ap.x * ab.x + ap.z * ab.z) / (ab.x * ab.x + ab.z * ab.z + 1e-9);
    let t = t.clamp(0.0, 1.0);
    a.lerp(&b, t)
}

// ---------------------------------------------------------------------------
// Navigation flags
// ---------------------------------------------------------------------------

/// Bitflags for navigation polygon properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NavFlags(u32);

impl NavFlags {
    pub const NONE: Self = Self(0);
    pub const WALKABLE: Self = Self(1);
    pub const SWIMABLE: Self = Self(2);
    pub const CLIMBABLE: Self = Self(4);
    pub const DANGEROUS: Self = Self(8);
    pub const RESTRICTED: Self = Self(16);
    pub const INDOOR: Self = Self(32);
    pub const OUTDOOR: Self = Self(64);

    pub fn has(&self, flag: Self) -> bool { (self.0 & flag.0) != 0 }
    pub fn set(&mut self, flag: Self) { self.0 |= flag.0; }
    pub fn clear(&mut self, flag: Self) { self.0 &= !flag.0; }
}

// ---------------------------------------------------------------------------
// Off-mesh link
// ---------------------------------------------------------------------------

/// Type of off-mesh link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkType {
    /// Standard jump link.
    Jump,
    /// Ladder (climb up/down).
    Ladder,
    /// Teleporter (instant transport).
    Teleporter,
    /// Door (may be locked/unlocked).
    Door,
    /// Drop-down (one way, fall).
    DropDown,
    /// Zip-line.
    ZipLine,
    /// Custom link type.
    Custom(u32),
}

/// An off-mesh link connecting two navmesh polygons.
#[derive(Debug, Clone)]
pub struct OffMeshLink {
    /// Unique link id.
    pub id: LinkId,
    /// Start position.
    pub start: Vec3,
    /// End position.
    pub end: Vec3,
    /// Start polygon (optional -- auto-detected if None).
    pub start_poly: Option<PolyId>,
    /// End polygon (optional).
    pub end_poly: Option<PolyId>,
    /// Link type.
    pub link_type: LinkType,
    /// Whether the link is bidirectional.
    pub bidirectional: bool,
    /// Whether the link is currently enabled.
    pub enabled: bool,
    /// Traversal cost.
    pub cost: f32,
    /// Traversal time in seconds.
    pub traversal_time: f32,
    /// Required agent radius (max radius that can use this link).
    pub max_agent_radius: f32,
    /// User-defined tag for filtering.
    pub tag: String,
}

impl OffMeshLink {
    /// Creates a jump link.
    pub fn jump(start: Vec3, end: Vec3, cost: f32) -> Self {
        Self {
            id: LinkId::new(),
            start,
            end,
            start_poly: None,
            end_poly: None,
            link_type: LinkType::Jump,
            bidirectional: false,
            enabled: true,
            cost,
            traversal_time: 0.5,
            max_agent_radius: 0.5,
            tag: String::new(),
        }
    }

    /// Creates a ladder link.
    pub fn ladder(bottom: Vec3, top: Vec3, climb_time: f32) -> Self {
        Self {
            id: LinkId::new(),
            start: bottom,
            end: top,
            start_poly: None,
            end_poly: None,
            link_type: LinkType::Ladder,
            bidirectional: true,
            enabled: true,
            cost: climb_time * 2.0,
            traversal_time: climb_time,
            max_agent_radius: 0.4,
            tag: String::new(),
        }
    }

    /// Creates a teleporter link.
    pub fn teleporter(entry: Vec3, exit: Vec3) -> Self {
        Self {
            id: LinkId::new(),
            start: entry,
            end: exit,
            start_poly: None,
            end_poly: None,
            link_type: LinkType::Teleporter,
            bidirectional: false,
            enabled: true,
            cost: 0.1,
            traversal_time: 0.0,
            max_agent_radius: 1.0,
            tag: String::new(),
        }
    }

    /// Creates a drop-down link (one-way fall).
    pub fn drop_down(top: Vec3, bottom: Vec3) -> Self {
        Self {
            id: LinkId::new(),
            start: top,
            end: bottom,
            start_poly: None,
            end_poly: None,
            link_type: LinkType::DropDown,
            bidirectional: false,
            enabled: true,
            cost: 1.0,
            traversal_time: 0.3,
            max_agent_radius: 0.5,
            tag: String::new(),
        }
    }

    /// Creates a door link.
    pub fn door(side_a: Vec3, side_b: Vec3) -> Self {
        Self {
            id: LinkId::new(),
            start: side_a,
            end: side_b,
            start_poly: None,
            end_poly: None,
            link_type: LinkType::Door,
            bidirectional: true,
            enabled: true,
            cost: 1.5,
            traversal_time: 1.0,
            max_agent_radius: 0.5,
            tag: String::new(),
        }
    }

    /// Returns the length of the link.
    pub fn length(&self) -> f32 {
        self.start.distance(&self.end)
    }
}

// ---------------------------------------------------------------------------
// Dynamic obstacle
// ---------------------------------------------------------------------------

/// A dynamic obstacle that modifies the navmesh at runtime.
#[derive(Debug, Clone)]
pub struct DynamicObstacle {
    /// Unique obstacle id.
    pub id: ObstacleId,
    /// Center position.
    pub position: Vec3,
    /// Obstacle radius.
    pub radius: f32,
    /// Obstacle height.
    pub height: f32,
    /// Whether the obstacle is active.
    pub active: bool,
    /// Affected polygon ids (computed on registration).
    pub affected_polys: Vec<PolyId>,
}

impl DynamicObstacle {
    /// Creates a cylindrical obstacle.
    pub fn cylinder(position: Vec3, radius: f32, height: f32) -> Self {
        Self {
            id: ObstacleId::new(),
            position,
            radius,
            height,
            active: true,
            affected_polys: Vec::new(),
        }
    }

    /// Whether a point is inside the obstacle (XZ plane).
    pub fn contains_point_xz(&self, p: Vec3) -> bool {
        self.position.distance_xz(&p) <= self.radius
    }
}

// ---------------------------------------------------------------------------
// Path result
// ---------------------------------------------------------------------------

/// A navigation path consisting of waypoints.
#[derive(Debug, Clone)]
pub struct NavPath {
    /// Ordered waypoints from start to goal.
    pub waypoints: Vec<Vec3>,
    /// Polygon ids along the path.
    pub poly_ids: Vec<PolyId>,
    /// Off-mesh links encountered along the path.
    pub links: Vec<LinkId>,
    /// Total path length.
    pub total_length: f32,
    /// Total estimated traversal cost.
    pub total_cost: f32,
    /// Whether the path reaches the goal.
    pub complete: bool,
    /// Whether the path is still valid.
    pub valid: bool,
}

impl NavPath {
    /// Creates an empty (failed) path.
    pub fn failed() -> Self {
        Self {
            waypoints: Vec::new(),
            poly_ids: Vec::new(),
            links: Vec::new(),
            total_length: 0.0,
            total_cost: 0.0,
            complete: false,
            valid: false,
        }
    }

    /// Returns the number of waypoints.
    pub fn waypoint_count(&self) -> usize {
        self.waypoints.len()
    }

    /// Returns true if the path has waypoints.
    pub fn is_valid(&self) -> bool {
        self.valid && !self.waypoints.is_empty()
    }

    /// Returns the waypoint at a given index.
    pub fn waypoint(&self, index: usize) -> Option<&Vec3> {
        self.waypoints.get(index)
    }

    /// Computes the total length of the path.
    pub fn compute_length(&self) -> f32 {
        let mut len = 0.0;
        for i in 1..self.waypoints.len() {
            len += self.waypoints[i - 1].distance(&self.waypoints[i]);
        }
        len
    }

    /// Returns a position along the path at a normalized parameter t (0..1).
    pub fn sample(&self, t: f32) -> Option<Vec3> {
        if self.waypoints.is_empty() { return None; }
        if self.waypoints.len() == 1 { return Some(self.waypoints[0]); }

        let total = self.compute_length();
        let target_dist = t.clamp(0.0, 1.0) * total;

        let mut accumulated = 0.0;
        for i in 1..self.waypoints.len() {
            let seg_len = self.waypoints[i - 1].distance(&self.waypoints[i]);
            if accumulated + seg_len >= target_dist {
                let local_t = (target_dist - accumulated) / seg_len;
                return Some(self.waypoints[i - 1].lerp(&self.waypoints[i], local_t));
            }
            accumulated += seg_len;
        }

        Some(*self.waypoints.last().unwrap())
    }
}

// ---------------------------------------------------------------------------
// Path corridor
// ---------------------------------------------------------------------------

/// A path corridor for smooth path following with dynamic re-planning.
#[derive(Debug, Clone)]
pub struct PathCorridor {
    /// The current path.
    path: NavPath,
    /// Current position along the corridor.
    current_position: Vec3,
    /// Current waypoint index.
    current_waypoint: usize,
    /// Target position (final destination).
    target: Vec3,
    /// Corridor width (how far from the path the agent can deviate).
    corridor_width: f32,
    /// Look-ahead distance for smooth steering.
    look_ahead: f32,
    /// Whether the agent has reached the goal.
    reached_goal: bool,
    /// Goal threshold distance.
    goal_threshold: f32,
}

impl PathCorridor {
    /// Creates a new path corridor.
    pub fn new(path: NavPath, corridor_width: f32) -> Self {
        let target = path.waypoints.last().copied().unwrap_or(Vec3::ZERO);
        Self {
            path,
            current_position: Vec3::ZERO,
            current_waypoint: 0,
            target,
            corridor_width,
            look_ahead: 2.0,
            reached_goal: false,
            goal_threshold: 0.5,
        }
    }

    /// Updates the corridor with the agent's current position.
    pub fn update(&mut self, agent_position: Vec3) {
        self.current_position = agent_position;

        // Advance waypoints.
        while self.current_waypoint < self.path.waypoints.len() {
            let wp = self.path.waypoints[self.current_waypoint];
            if agent_position.distance_xz(&wp) < self.goal_threshold {
                self.current_waypoint += 1;
            } else {
                break;
            }
        }

        // Check if goal reached.
        if agent_position.distance_xz(&self.target) < self.goal_threshold {
            self.reached_goal = true;
        }
    }

    /// Returns the next steering target position.
    pub fn steering_target(&self) -> Vec3 {
        if self.reached_goal {
            return self.target;
        }

        // Find the look-ahead point.
        let mut dist = 0.0;
        for i in self.current_waypoint..self.path.waypoints.len() {
            let wp = self.path.waypoints[i];
            let seg_dist = if i == self.current_waypoint {
                self.current_position.distance_xz(&wp)
            } else {
                self.path.waypoints[i - 1].distance_xz(&wp)
            };

            dist += seg_dist;
            if dist >= self.look_ahead {
                return wp;
            }
        }

        self.target
    }

    /// Returns whether the agent has reached the goal.
    pub fn is_at_goal(&self) -> bool {
        self.reached_goal
    }

    /// Returns the current waypoint index.
    pub fn current_waypoint_index(&self) -> usize {
        self.current_waypoint
    }

    /// Returns the remaining distance to the goal.
    pub fn remaining_distance(&self) -> f32 {
        if self.reached_goal { return 0.0; }
        let mut dist = self.current_position.distance_xz(
            self.path.waypoints.get(self.current_waypoint)
                .unwrap_or(&self.target),
        );
        for i in (self.current_waypoint + 1)..self.path.waypoints.len() {
            dist += self.path.waypoints[i - 1].distance_xz(&self.path.waypoints[i]);
        }
        dist
    }

    /// Whether the path needs replanning (agent deviated too far).
    pub fn needs_replan(&self) -> bool {
        if self.path.waypoints.is_empty() || self.reached_goal {
            return false;
        }

        if self.current_waypoint >= self.path.waypoints.len() {
            return false;
        }

        // Check if the agent is too far from the corridor.
        let wp = self.path.waypoints[self.current_waypoint];
        let deviation = self.current_position.distance_xz(&wp);
        deviation > self.corridor_width
    }

    /// Replaces the path (after replanning).
    pub fn set_path(&mut self, path: NavPath) {
        self.target = path.waypoints.last().copied().unwrap_or(self.target);
        self.path = path;
        self.current_waypoint = 0;
        self.reached_goal = false;
    }

    /// Sets the look-ahead distance.
    pub fn set_look_ahead(&mut self, distance: f32) {
        self.look_ahead = distance.max(0.1);
    }

    /// Sets the goal threshold distance.
    pub fn set_goal_threshold(&mut self, threshold: f32) {
        self.goal_threshold = threshold.max(0.01);
    }
}

// ---------------------------------------------------------------------------
// Query cache
// ---------------------------------------------------------------------------

/// Cache key for navigation queries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QueryKey {
    start: (i32, i32, i32),
    end: (i32, i32, i32),
}

impl QueryKey {
    fn from_positions(start: Vec3, end: Vec3, precision: f32) -> Self {
        Self {
            start: (
                (start.x / precision).round() as i32,
                (start.y / precision).round() as i32,
                (start.z / precision).round() as i32,
            ),
            end: (
                (end.x / precision).round() as i32,
                (end.y / precision).round() as i32,
                (end.z / precision).round() as i32,
            ),
        }
    }
}

/// LRU cache for navigation path queries.
struct NavQueryCache {
    entries: HashMap<QueryKey, NavPath>,
    order: VecDeque<QueryKey>,
    max_size: usize,
    cache_precision: f32,
    hits: u64,
    misses: u64,
}

impl NavQueryCache {
    fn new(max_size: usize, precision: f32) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_size,
            cache_precision: precision,
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, start: Vec3, end: Vec3) -> Option<&NavPath> {
        let key = QueryKey::from_positions(start, end, self.cache_precision);
        if self.entries.contains_key(&key) {
            self.hits += 1;
            // Move to back of LRU.
            self.order.retain(|k| k != &key);
            self.order.push_back(key.clone());
            self.entries.get(&key)
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, start: Vec3, end: Vec3, path: NavPath) {
        let key = QueryKey::from_positions(start, end, self.cache_precision);

        // Evict if at capacity.
        while self.entries.len() >= self.max_size {
            if let Some(old_key) = self.order.pop_front() {
                self.entries.remove(&old_key);
            }
        }

        self.order.push_back(key.clone());
        self.entries.insert(key, path);
    }

    fn invalidate_all(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    fn invalidate_near(&mut self, position: Vec3, radius: f32) {
        let to_remove: Vec<QueryKey> = self.entries.keys()
            .filter(|k| {
                let start = Vec3::new(
                    k.start.0 as f32 * self.cache_precision,
                    k.start.1 as f32 * self.cache_precision,
                    k.start.2 as f32 * self.cache_precision,
                );
                start.distance(&position) < radius
            })
            .cloned()
            .collect();

        for key in &to_remove {
            self.entries.remove(key);
            self.order.retain(|k| k != key);
        }
    }

    fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f32 / total as f32 }
    }

    fn size(&self) -> usize {
        self.entries.len()
    }
}

impl fmt::Debug for NavQueryCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NavQueryCache")
            .field("size", &self.entries.len())
            .field("max_size", &self.max_size)
            .field("hit_rate", &self.hit_rate())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// A* node for pathfinding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct AStarNode {
    poly_id: PolyId,
    g_cost: f32,
    f_cost: f32,
    parent: Option<PolyId>,
    via_link: Option<LinkId>,
}

impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool { self.f_cost == other.f_cost }
}
impl Eq for AStarNode {}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap.
        other.f_cost.partial_cmp(&self.f_cost).unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Navigation manager
// ---------------------------------------------------------------------------

/// Configuration for the navigation manager.
#[derive(Debug, Clone)]
pub struct NavConfig {
    /// Cell size in world units.
    pub cell_size: f32,
    /// Load radius in cells.
    pub load_radius: u32,
    /// Unload radius in cells.
    pub unload_radius: u32,
    /// Maximum path query distance.
    pub max_path_distance: f32,
    /// Query cache size.
    pub cache_size: usize,
    /// Cache spatial precision.
    pub cache_precision: f32,
    /// Default agent radius.
    pub agent_radius: f32,
    /// Default agent height.
    pub agent_height: f32,
}

impl Default for NavConfig {
    fn default() -> Self {
        Self {
            cell_size: 64.0,
            load_radius: 4,
            unload_radius: 6,
            max_path_distance: 1000.0,
            cache_size: 256,
            cache_precision: 1.0,
            agent_radius: 0.5,
            agent_height: 2.0,
        }
    }
}

/// The world-scale navigation manager.
pub struct NavigationManager {
    /// All loaded navmesh polygons.
    polygons: HashMap<PolyId, NavPoly>,
    /// Polygons per cell.
    cell_polys: HashMap<CellCoord, Vec<PolyId>>,
    /// Off-mesh links.
    links: HashMap<LinkId, OffMeshLink>,
    /// Dynamic obstacles.
    obstacles: HashMap<ObstacleId, DynamicObstacle>,
    /// Navigation config.
    config: NavConfig,
    /// Query cache.
    cache: NavQueryCache,
    /// Loaded cells.
    loaded_cells: HashSet<CellCoord>,
    /// Current viewer position.
    viewer_position: Vec3,
    /// Current viewer cell.
    viewer_cell: CellCoord,
}

impl NavigationManager {
    /// Creates a new navigation manager.
    pub fn new(config: NavConfig) -> Self {
        let cache = NavQueryCache::new(config.cache_size, config.cache_precision);
        Self {
            polygons: HashMap::new(),
            cell_polys: HashMap::new(),
            links: HashMap::new(),
            obstacles: HashMap::new(),
            config,
            cache,
            loaded_cells: HashSet::new(),
            viewer_position: Vec3::ZERO,
            viewer_cell: CellCoord::new(0, 0),
        }
    }

    // -- Polygon management ---------------------------------------------------

    /// Registers a navmesh polygon.
    pub fn add_polygon(&mut self, poly: NavPoly) -> PolyId {
        let id = poly.id;
        let cell = poly.cell;
        self.cell_polys.entry(cell).or_insert_with(Vec::new).push(id);
        self.polygons.insert(id, poly);
        id
    }

    /// Removes a polygon.
    pub fn remove_polygon(&mut self, id: PolyId) {
        if let Some(poly) = self.polygons.remove(&id) {
            if let Some(cell_polys) = self.cell_polys.get_mut(&poly.cell) {
                cell_polys.retain(|p| *p != id);
            }
        }
    }

    /// Returns a polygon by id.
    pub fn get_polygon(&self, id: PolyId) -> Option<&NavPoly> {
        self.polygons.get(&id)
    }

    /// Returns the total polygon count.
    pub fn polygon_count(&self) -> usize {
        self.polygons.len()
    }

    /// Finds the polygon containing a point (XZ plane).
    pub fn find_polygon_at(&self, position: Vec3) -> Option<PolyId> {
        let cell = self.world_to_cell(position);

        // Search in this cell and neighbors.
        let search_cells = [
            cell,
            CellCoord::new(cell.x - 1, cell.z),
            CellCoord::new(cell.x + 1, cell.z),
            CellCoord::new(cell.x, cell.z - 1),
            CellCoord::new(cell.x, cell.z + 1),
        ];

        for search_cell in &search_cells {
            if let Some(poly_ids) = self.cell_polys.get(search_cell) {
                for &pid in poly_ids {
                    if let Some(poly) = self.polygons.get(&pid) {
                        if poly.contains_point_xz(position) {
                            return Some(pid);
                        }
                    }
                }
            }
        }

        None
    }

    /// Finds the nearest polygon to a position.
    pub fn find_nearest_polygon(&self, position: Vec3, search_radius: f32) -> Option<PolyId> {
        let mut best_id = None;
        let mut best_dist = search_radius;

        for poly in self.polygons.values() {
            let dist = position.distance_xz(&poly.center);
            if dist < best_dist {
                best_dist = dist;
                best_id = Some(poly.id);
            }
        }

        best_id
    }

    // -- Link management ------------------------------------------------------

    /// Registers an off-mesh link.
    pub fn add_link(&mut self, link: OffMeshLink) -> LinkId {
        let id = link.id;
        self.links.insert(id, link);
        self.cache.invalidate_all();
        id
    }

    /// Removes a link.
    pub fn remove_link(&mut self, id: LinkId) {
        self.links.remove(&id);
        self.cache.invalidate_all();
    }

    /// Returns a link by id.
    pub fn get_link(&self, id: LinkId) -> Option<&OffMeshLink> {
        self.links.get(&id)
    }

    /// Enables or disables a link (e.g., lock/unlock a door).
    pub fn set_link_enabled(&mut self, id: LinkId, enabled: bool) {
        if let Some(link) = self.links.get_mut(&id) {
            link.enabled = enabled;
            self.cache.invalidate_all();
        }
    }

    /// Returns the link count.
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    // -- Obstacle management --------------------------------------------------

    /// Registers a dynamic obstacle.
    pub fn add_obstacle(&mut self, obstacle: DynamicObstacle) -> ObstacleId {
        let id = obstacle.id;
        self.cache.invalidate_near(obstacle.position, obstacle.radius * 2.0);
        self.obstacles.insert(id, obstacle);
        id
    }

    /// Removes an obstacle.
    pub fn remove_obstacle(&mut self, id: ObstacleId) {
        if let Some(obstacle) = self.obstacles.remove(&id) {
            self.cache.invalidate_near(obstacle.position, obstacle.radius * 2.0);
        }
    }

    /// Moves an obstacle.
    pub fn move_obstacle(&mut self, id: ObstacleId, new_position: Vec3) {
        if let Some(obstacle) = self.obstacles.get_mut(&id) {
            self.cache.invalidate_near(obstacle.position, obstacle.radius * 2.0);
            obstacle.position = new_position;
            self.cache.invalidate_near(new_position, obstacle.radius * 2.0);
        }
    }

    /// Returns the obstacle count.
    pub fn obstacle_count(&self) -> usize {
        self.obstacles.len()
    }

    // -- Pathfinding ----------------------------------------------------------

    /// Finds a path from start to goal using A*.
    pub fn find_path(&mut self, start: Vec3, goal: Vec3) -> NavPath {
        // Check cache first.
        if let Some(cached) = self.cache.get(start, goal) {
            return cached.clone();
        }

        // Find start and goal polygons.
        let start_poly = match self.find_nearest_polygon(start, 10.0) {
            Some(id) => id,
            None => return NavPath::failed(),
        };
        let goal_poly = match self.find_nearest_polygon(goal, 10.0) {
            Some(id) => id,
            None => return NavPath::failed(),
        };

        if start_poly == goal_poly {
            let path = NavPath {
                waypoints: vec![start, goal],
                poly_ids: vec![start_poly],
                links: Vec::new(),
                total_length: start.distance(&goal),
                total_cost: start.distance(&goal),
                complete: true,
                valid: true,
            };
            self.cache.insert(start, goal, path.clone());
            return path;
        }

        // A* search.
        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashSet::new();
        let mut came_from: HashMap<PolyId, (Option<PolyId>, Option<LinkId>)> = HashMap::new();
        let mut g_scores: HashMap<PolyId, f32> = HashMap::new();

        let start_center = self.polygons.get(&start_poly)
            .map(|p| p.center)
            .unwrap_or(start);
        let goal_center = self.polygons.get(&goal_poly)
            .map(|p| p.center)
            .unwrap_or(goal);

        g_scores.insert(start_poly, 0.0);
        open_set.push(AStarNode {
            poly_id: start_poly,
            g_cost: 0.0,
            f_cost: start_center.distance(&goal_center),
            parent: None,
            via_link: None,
        });

        while let Some(current) = open_set.pop() {
            if current.poly_id == goal_poly {
                // Reconstruct path.
                let path = self.reconstruct_path(start, goal, &came_from, goal_poly);
                self.cache.insert(start, goal, path.clone());
                return path;
            }

            if !closed_set.insert(current.poly_id) {
                continue;
            }

            // Expand neighbors.
            if let Some(poly) = self.polygons.get(&current.poly_id) {
                let neighbors: Vec<PolyId> = poly.neighbors.iter()
                    .filter_map(|n| *n)
                    .collect();

                for neighbor_id in neighbors {
                    if closed_set.contains(&neighbor_id) {
                        continue;
                    }

                    // Check obstacles.
                    if let Some(npoly) = self.polygons.get(&neighbor_id) {
                        let blocked = self.obstacles.values().any(|obs| {
                            obs.active && obs.contains_point_xz(npoly.center)
                        });
                        if blocked {
                            continue;
                        }

                        let move_cost = poly.center.distance(&npoly.center) * npoly.cost;
                        let new_g = current.g_cost + move_cost;

                        if new_g < *g_scores.get(&neighbor_id).unwrap_or(&f32::MAX) {
                            g_scores.insert(neighbor_id, new_g);
                            came_from.insert(neighbor_id, (Some(current.poly_id), None));

                            let h = npoly.center.distance(&goal_center);
                            open_set.push(AStarNode {
                                poly_id: neighbor_id,
                                g_cost: new_g,
                                f_cost: new_g + h,
                                parent: Some(current.poly_id),
                                via_link: None,
                            });
                        }
                    }
                }
            }

            // Check off-mesh links from current polygon.
            for link in self.links.values() {
                if !link.enabled {
                    continue;
                }

                let is_start_here = self.find_nearest_polygon(link.start, 2.0) == Some(current.poly_id);
                let is_end_here = link.bidirectional
                    && self.find_nearest_polygon(link.end, 2.0) == Some(current.poly_id);

                let (link_target_pos, _link_start_pos) = if is_start_here {
                    (link.end, link.start)
                } else if is_end_here {
                    (link.start, link.end)
                } else {
                    continue;
                };

                if let Some(target_poly) = self.find_nearest_polygon(link_target_pos, 2.0) {
                    if closed_set.contains(&target_poly) {
                        continue;
                    }

                    let new_g = current.g_cost + link.cost;
                    if new_g < *g_scores.get(&target_poly).unwrap_or(&f32::MAX) {
                        g_scores.insert(target_poly, new_g);
                        came_from.insert(target_poly, (Some(current.poly_id), Some(link.id)));

                        if let Some(tpoly) = self.polygons.get(&target_poly) {
                            let h = tpoly.center.distance(&goal_center);
                            open_set.push(AStarNode {
                                poly_id: target_poly,
                                g_cost: new_g,
                                f_cost: new_g + h,
                                parent: Some(current.poly_id),
                                via_link: Some(link.id),
                            });
                        }
                    }
                }
            }
        }

        NavPath::failed()
    }

    /// Reconstructs a path from the A* came_from map.
    fn reconstruct_path(
        &self,
        start: Vec3,
        goal: Vec3,
        came_from: &HashMap<PolyId, (Option<PolyId>, Option<LinkId>)>,
        goal_poly: PolyId,
    ) -> NavPath {
        let mut poly_ids = Vec::new();
        let mut links = Vec::new();
        let mut current = goal_poly;

        poly_ids.push(current);

        while let Some((parent, link)) = came_from.get(&current) {
            if let Some(link_id) = link {
                links.push(*link_id);
            }
            match parent {
                Some(p) => {
                    poly_ids.push(*p);
                    current = *p;
                }
                None => break,
            }
        }

        poly_ids.reverse();
        links.reverse();

        // Build waypoints from polygon centers.
        let mut waypoints = vec![start];
        for &pid in &poly_ids {
            if let Some(poly) = self.polygons.get(&pid) {
                waypoints.push(poly.center);
            }
        }
        waypoints.push(goal);

        let total_length = {
            let mut len = 0.0;
            for i in 1..waypoints.len() {
                len += waypoints[i - 1].distance(&waypoints[i]);
            }
            len
        };

        NavPath {
            waypoints,
            poly_ids,
            links,
            total_length,
            total_cost: total_length,
            complete: true,
            valid: true,
        }
    }

    // -- Cell streaming -------------------------------------------------------

    /// Converts a world position to a cell coordinate.
    pub fn world_to_cell(&self, pos: Vec3) -> CellCoord {
        CellCoord {
            x: (pos.x / self.config.cell_size).floor() as i32,
            z: (pos.z / self.config.cell_size).floor() as i32,
        }
    }

    /// Loads navmesh data for a cell.
    pub fn load_cell(&mut self, coord: CellCoord) {
        self.loaded_cells.insert(coord);
    }

    /// Unloads navmesh data for a cell.
    pub fn unload_cell(&mut self, coord: CellCoord) {
        self.loaded_cells.remove(&coord);
        // Remove polygons in this cell.
        if let Some(poly_ids) = self.cell_polys.remove(&coord) {
            for pid in poly_ids {
                self.polygons.remove(&pid);
            }
        }
        self.cache.invalidate_all();
    }

    /// Returns loaded cell count.
    pub fn loaded_cell_count(&self) -> usize {
        self.loaded_cells.len()
    }

    // -- Per-frame update -----------------------------------------------------

    /// Updates the navigation manager with the viewer position.
    pub fn update(&mut self, viewer_position: Vec3) {
        self.viewer_position = viewer_position;
        self.viewer_cell = self.world_to_cell(viewer_position);
    }

    // -- Cache stats ----------------------------------------------------------

    /// Returns cache hit rate.
    pub fn cache_hit_rate(&self) -> f32 {
        self.cache.hit_rate()
    }

    /// Returns cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.size()
    }

    /// Invalidates the entire cache.
    pub fn invalidate_cache(&mut self) {
        self.cache.invalidate_all();
    }
}

impl Default for NavigationManager {
    fn default() -> Self {
        Self::new(NavConfig::default())
    }
}

impl fmt::Debug for NavigationManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NavigationManager")
            .field("polygons", &self.polygons.len())
            .field("links", &self.links.len())
            .field("obstacles", &self.obstacles.len())
            .field("loaded_cells", &self.loaded_cells.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quad_poly(id: PolyId, x: f32, z: f32, size: f32) -> NavPoly {
        NavPoly::new(
            id,
            vec![
                Vec3::new(x, 0.0, z),
                Vec3::new(x + size, 0.0, z),
                Vec3::new(x + size, 0.0, z + size),
                Vec3::new(x, 0.0, z + size),
            ],
            CellCoord::new(0, 0),
        )
    }

    fn make_connected_mesh() -> NavigationManager {
        let mut nav = NavigationManager::new(NavConfig::default());

        let id_a = PolyId::new();
        let id_b = PolyId::new();
        let id_c = PolyId::new();

        let mut poly_a = make_quad_poly(id_a, 0.0, 0.0, 10.0);
        let mut poly_b = make_quad_poly(id_b, 10.0, 0.0, 10.0);
        let mut poly_c = make_quad_poly(id_c, 20.0, 0.0, 10.0);

        // Connect A <-> B <-> C.
        poly_a.neighbors[1] = Some(id_b); // East edge
        poly_b.neighbors[3] = Some(id_a); // West edge
        poly_b.neighbors[1] = Some(id_c); // East edge
        poly_c.neighbors[3] = Some(id_b); // West edge

        nav.add_polygon(poly_a);
        nav.add_polygon(poly_b);
        nav.add_polygon(poly_c);

        nav
    }

    #[test]
    fn polygon_contains_point() {
        let poly = make_quad_poly(PolyId::new(), 0.0, 0.0, 10.0);
        assert!(poly.contains_point_xz(Vec3::new(5.0, 0.0, 5.0)));
        assert!(!poly.contains_point_xz(Vec3::new(15.0, 0.0, 5.0)));
    }

    #[test]
    fn polygon_area() {
        let poly = make_quad_poly(PolyId::new(), 0.0, 0.0, 10.0);
        assert!((poly.area - 100.0).abs() < 1.0);
    }

    #[test]
    fn find_polygon_at() {
        let mut nav = NavigationManager::new(NavConfig::default());
        let poly = make_quad_poly(PolyId::new(), 0.0, 0.0, 10.0);
        let pid = nav.add_polygon(poly);

        assert_eq!(nav.find_polygon_at(Vec3::new(5.0, 0.0, 5.0)), Some(pid));
        assert_eq!(nav.find_polygon_at(Vec3::new(50.0, 0.0, 50.0)), None);
    }

    #[test]
    fn find_path_same_poly() {
        let mut nav = NavigationManager::new(NavConfig::default());
        let poly = make_quad_poly(PolyId::new(), 0.0, 0.0, 20.0);
        nav.add_polygon(poly);

        let path = nav.find_path(Vec3::new(2.0, 0.0, 2.0), Vec3::new(8.0, 0.0, 8.0));
        assert!(path.is_valid());
        assert!(path.complete);
    }

    #[test]
    fn find_path_across_polys() {
        let mut nav = make_connected_mesh();
        let path = nav.find_path(Vec3::new(5.0, 0.0, 5.0), Vec3::new(25.0, 0.0, 5.0));
        assert!(path.is_valid());
        assert!(path.waypoint_count() >= 2);
    }

    #[test]
    fn path_failed_no_navmesh() {
        let mut nav = NavigationManager::new(NavConfig::default());
        let path = nav.find_path(Vec3::new(0.0, 0.0, 0.0), Vec3::new(100.0, 0.0, 100.0));
        assert!(!path.is_valid());
    }

    #[test]
    fn off_mesh_link_jump() {
        let mut nav = make_connected_mesh();
        let link = OffMeshLink::jump(
            Vec3::new(25.0, 0.0, 5.0),
            Vec3::new(40.0, 0.0, 5.0),
            2.0,
        );
        nav.add_link(link);

        assert_eq!(nav.link_count(), 1);
    }

    #[test]
    fn off_mesh_link_ladder() {
        let link = OffMeshLink::ladder(
            Vec3::new(5.0, 0.0, 5.0),
            Vec3::new(5.0, 10.0, 5.0),
            3.0,
        );
        assert!(link.bidirectional);
        assert_eq!(link.link_type, LinkType::Ladder);
    }

    #[test]
    fn off_mesh_link_teleporter() {
        let link = OffMeshLink::teleporter(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 100.0),
        );
        assert!(!link.bidirectional);
        assert!((link.cost - 0.1).abs() < 0.01);
    }

    #[test]
    fn dynamic_obstacle() {
        let mut nav = make_connected_mesh();
        let obs = DynamicObstacle::cylinder(Vec3::new(15.0, 0.0, 5.0), 3.0, 2.0);
        let obs_id = nav.add_obstacle(obs);

        assert_eq!(nav.obstacle_count(), 1);

        nav.move_obstacle(obs_id, Vec3::new(20.0, 0.0, 5.0));
        nav.remove_obstacle(obs_id);
        assert_eq!(nav.obstacle_count(), 0);
    }

    #[test]
    fn path_corridor_basic() {
        let path = NavPath {
            waypoints: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(10.0, 0.0, 0.0),
                Vec3::new(20.0, 0.0, 0.0),
            ],
            poly_ids: Vec::new(),
            links: Vec::new(),
            total_length: 20.0,
            total_cost: 20.0,
            complete: true,
            valid: true,
        };

        let mut corridor = PathCorridor::new(path, 3.0);
        corridor.update(Vec3::new(0.0, 0.0, 0.0));

        let target = corridor.steering_target();
        assert!(target.x > 0.0);
        assert!(!corridor.is_at_goal());
    }

    #[test]
    fn path_corridor_reaches_goal() {
        let path = NavPath {
            waypoints: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(5.0, 0.0, 0.0),
            ],
            poly_ids: Vec::new(),
            links: Vec::new(),
            total_length: 5.0,
            total_cost: 5.0,
            complete: true,
            valid: true,
        };

        let mut corridor = PathCorridor::new(path, 3.0);
        corridor.update(Vec3::new(5.0, 0.0, 0.0));
        assert!(corridor.is_at_goal());
    }

    #[test]
    fn path_corridor_replan_needed() {
        let path = NavPath {
            waypoints: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(10.0, 0.0, 0.0),
            ],
            poly_ids: Vec::new(),
            links: Vec::new(),
            total_length: 10.0,
            total_cost: 10.0,
            complete: true,
            valid: true,
        };

        let mut corridor = PathCorridor::new(path, 2.0);
        corridor.update(Vec3::new(0.0, 0.0, 10.0)); // Far off path
        assert!(corridor.needs_replan());
    }

    #[test]
    fn path_sampling() {
        let path = NavPath {
            waypoints: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(10.0, 0.0, 0.0),
            ],
            poly_ids: Vec::new(),
            links: Vec::new(),
            total_length: 10.0,
            total_cost: 10.0,
            complete: true,
            valid: true,
        };

        let mid = path.sample(0.5).unwrap();
        assert!((mid.x - 5.0).abs() < 0.1);
    }

    #[test]
    fn cache_operations() {
        let mut cache = NavQueryCache::new(10, 1.0);

        let path = NavPath {
            waypoints: vec![Vec3::ZERO],
            poly_ids: Vec::new(),
            links: Vec::new(),
            total_length: 0.0,
            total_cost: 0.0,
            complete: true,
            valid: true,
        };

        let start = Vec3::new(0.0, 0.0, 0.0);
        let end = Vec3::new(10.0, 0.0, 10.0);

        cache.insert(start, end, path);
        assert_eq!(cache.size(), 1);

        let result = cache.get(start, end);
        assert!(result.is_some());
        assert!(cache.hit_rate() > 0.0);
    }

    #[test]
    fn nav_flags() {
        let mut flags = NavFlags::WALKABLE;
        flags.set(NavFlags::OUTDOOR);
        assert!(flags.has(NavFlags::WALKABLE));
        assert!(flags.has(NavFlags::OUTDOOR));
        assert!(!flags.has(NavFlags::SWIMABLE));

        flags.clear(NavFlags::WALKABLE);
        assert!(!flags.has(NavFlags::WALKABLE));
    }

    #[test]
    fn cell_streaming() {
        let mut nav = NavigationManager::new(NavConfig::default());
        nav.load_cell(CellCoord::new(0, 0));
        assert_eq!(nav.loaded_cell_count(), 1);
        nav.unload_cell(CellCoord::new(0, 0));
        assert_eq!(nav.loaded_cell_count(), 0);
    }

    #[test]
    fn link_enable_disable() {
        let mut nav = NavigationManager::new(NavConfig::default());
        let link = OffMeshLink::door(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0));
        let id = nav.add_link(link);

        assert!(nav.get_link(id).unwrap().enabled);
        nav.set_link_enabled(id, false);
        assert!(!nav.get_link(id).unwrap().enabled);
    }

    #[test]
    fn world_to_cell() {
        let nav = NavigationManager::new(NavConfig {
            cell_size: 64.0,
            ..NavConfig::default()
        });
        let cell = nav.world_to_cell(Vec3::new(100.0, 0.0, 200.0));
        assert_eq!(cell.x, 1);
        assert_eq!(cell.z, 3);
    }
}
