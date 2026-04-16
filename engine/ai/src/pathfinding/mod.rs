//! Pathfinding algorithms.
//!
//! Provides A* and hierarchical (HPA*) pathfinding over arbitrary graph
//! representations. The [`NavGraph`] trait abstracts the search space so
//! callers can plug in grids, navmeshes, or custom graphs without changing
//! the pathfinding logic.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use glam::Vec3;

use genovo_core::EngineResult;

// ---------------------------------------------------------------------------
// NodeId
// ---------------------------------------------------------------------------

/// Opaque node identifier used by all graph / pathfinder types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Create a node id from an integer.
    #[inline]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Return the raw integer value.
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// NavGraph trait
// ---------------------------------------------------------------------------

/// Abstraction over a navigation graph that can be searched with A*.
///
/// Implementations provide the neighbor expansion and heuristic cost estimate
/// required by the search algorithm.
pub trait NavGraph {
    /// Returns the neighbors of `node` along with the edge cost to each.
    fn neighbors(&self, node: NodeId) -> Vec<(NodeId, f32)>;

    /// Admissible heuristic: estimated cost from `a` to `b`.
    ///
    /// Must never over-estimate the true cost for A* to remain optimal.
    fn heuristic(&self, a: NodeId, b: NodeId) -> f32;

    /// Convert a node id to a world-space position (for path output).
    fn node_position(&self, node: NodeId) -> Vec3;
}

// ---------------------------------------------------------------------------
// PathRequest
// ---------------------------------------------------------------------------

/// A request to compute a path between two points.
#[derive(Debug, Clone)]
pub struct PathRequest {
    /// Starting position in world space.
    pub start: Vec3,
    /// Goal position in world space.
    pub end: Vec3,
    /// Radius of the navigating agent (for clearance checks).
    pub agent_radius: f32,
    /// Maximum slope the agent can traverse (in radians).
    pub max_slope: f32,
    /// Optional maximum search distance (to limit A* expansion).
    pub max_search_distance: Option<f32>,
}

impl PathRequest {
    /// Creates a basic path request between two points.
    pub fn new(start: Vec3, end: Vec3) -> Self {
        Self {
            start,
            end,
            agent_radius: 0.5,
            max_slope: std::f32::consts::FRAC_PI_4,
            max_search_distance: None,
        }
    }

    /// Sets the agent radius.
    pub fn with_agent_radius(mut self, radius: f32) -> Self {
        self.agent_radius = radius;
        self
    }

    /// Sets the maximum slope.
    pub fn with_max_slope(mut self, slope_radians: f32) -> Self {
        self.max_slope = slope_radians;
        self
    }

    /// Sets the maximum search distance.
    pub fn with_max_search_distance(mut self, distance: f32) -> Self {
        self.max_search_distance = Some(distance);
        self
    }
}

// ---------------------------------------------------------------------------
// Path
// ---------------------------------------------------------------------------

/// A computed path consisting of ordered nodes and a total cost.
#[derive(Debug, Clone)]
pub struct Path {
    /// Ordered list of waypoints from start to end.
    pub waypoints: Vec<Vec3>,
    /// Node ids corresponding to waypoints (parallel to `waypoints`).
    pub nodes: Vec<NodeId>,
    /// Total path cost (distance or weighted cost).
    pub cost: f32,
    /// Whether this is a partial path (goal was unreachable within budget).
    pub is_partial: bool,
}

impl Path {
    /// Creates a new empty path.
    pub fn empty() -> Self {
        Self {
            waypoints: Vec::new(),
            nodes: Vec::new(),
            cost: 0.0,
            is_partial: false,
        }
    }

    /// Creates a new complete path.
    pub fn new(waypoints: Vec<Vec3>, nodes: Vec<NodeId>, cost: f32) -> Self {
        Self {
            waypoints,
            nodes,
            cost,
            is_partial: false,
        }
    }

    /// Creates a new partial path.
    pub fn partial(waypoints: Vec<Vec3>, nodes: Vec<NodeId>, cost: f32) -> Self {
        Self {
            waypoints,
            nodes,
            cost,
            is_partial: true,
        }
    }

    /// Returns `true` if the path has no waypoints.
    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }

    /// Returns the number of waypoints.
    pub fn len(&self) -> usize {
        self.waypoints.len()
    }

    /// Returns the total Euclidean length of the path.
    pub fn length(&self) -> f32 {
        self.waypoints
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .sum()
    }

    /// Simplifies the path by removing redundant collinear waypoints using
    /// the Ramer-Douglas-Peucker algorithm.
    pub fn simplify(&mut self, epsilon: f32) {
        if self.waypoints.len() < 3 {
            return;
        }
        let keep = ramer_douglas_peucker(&self.waypoints, epsilon);
        let mut new_waypoints = Vec::with_capacity(keep.len());
        let mut new_nodes = Vec::with_capacity(keep.len());
        for &i in &keep {
            new_waypoints.push(self.waypoints[i]);
            if i < self.nodes.len() {
                new_nodes.push(self.nodes[i]);
            }
        }
        self.waypoints = new_waypoints;
        self.nodes = new_nodes;
    }
}

/// Ramer-Douglas-Peucker line simplification. Returns indices to keep.
fn ramer_douglas_peucker(points: &[Vec3], epsilon: f32) -> Vec<usize> {
    if points.len() < 2 {
        return (0..points.len()).collect();
    }
    let mut result = Vec::new();
    rdp_recursive(points, 0, points.len() - 1, epsilon, &mut result);
    result.push(points.len() - 1);
    result.sort_unstable();
    result.dedup();
    result
}

fn rdp_recursive(
    points: &[Vec3],
    start: usize,
    end: usize,
    epsilon: f32,
    result: &mut Vec<usize>,
) {
    result.push(start);
    if end <= start + 1 {
        return;
    }
    let line_start = points[start];
    let line_end = points[end];
    let line_dir = line_end - line_start;
    let line_len_sq = line_dir.length_squared();

    let mut max_dist = 0.0f32;
    let mut max_idx = start;

    for i in (start + 1)..end {
        let dist = if line_len_sq < 1e-12 {
            (points[i] - line_start).length()
        } else {
            let t = ((points[i] - line_start).dot(line_dir) / line_len_sq).clamp(0.0, 1.0);
            let proj = line_start + line_dir * t;
            (points[i] - proj).length()
        };
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        rdp_recursive(points, start, max_idx, epsilon, result);
        rdp_recursive(points, max_idx, end, epsilon, result);
    }
}

// ---------------------------------------------------------------------------
// PathNode (internal A* search node)
// ---------------------------------------------------------------------------

/// A node in the A* search graph.
#[derive(Debug, Clone)]
pub struct PathNode {
    /// World-space position of this node.
    pub position: Vec3,
    /// Cost from start to this node (g-cost).
    pub g_cost: f32,
    /// Estimated cost from this node to the goal (h-cost / heuristic).
    pub h_cost: f32,
    /// Index of the parent node in the search tree (for path reconstruction).
    pub parent: Option<usize>,
    /// Index of this node in the graph.
    pub index: usize,
}

impl PathNode {
    /// Total estimated cost (f = g + h).
    pub fn f_cost(&self) -> f32 {
        self.g_cost + self.h_cost
    }
}

impl PartialEq for PathNode {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for PathNode {}

impl PartialOrd for PathNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PathNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior in BinaryHeap.
        other
            .f_cost()
            .partial_cmp(&self.f_cost())
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// OpenSetEntry (min-heap wrapper)
// ---------------------------------------------------------------------------

/// Wrapper for the binary heap so we get a min-heap by f-cost.
#[derive(Debug, Clone)]
struct OpenSetEntry {
    node_id: NodeId,
    f_cost: f32,
}

impl PartialEq for OpenSetEntry {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for OpenSetEntry {}

impl PartialOrd for OpenSetEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OpenSetEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap: lower f_cost = higher priority.
        other
            .f_cost
            .partial_cmp(&self.f_cost)
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// PathFinder trait
// ---------------------------------------------------------------------------

/// Trait for pathfinding algorithm implementations.
pub trait PathFinder: Send + Sync {
    /// Computes a path from `request.start` to `request.end`.
    fn find_path(&self, request: &PathRequest) -> EngineResult<Path>;

    /// Returns `true` if there is any valid path between the two points.
    fn is_reachable(&self, start: Vec3, end: Vec3, agent_radius: f32) -> bool;
}

// ---------------------------------------------------------------------------
// AStarPathfinder
// ---------------------------------------------------------------------------

/// A* pathfinding implementation over an arbitrary [`NavGraph`].
///
/// The heuristic weight can be tuned: 1.0 gives optimal A*, values > 1.0
/// give greedy (faster but suboptimal) behaviour.
pub struct AStarPathfinder<G: NavGraph> {
    /// The navigation graph to search.
    pub graph: G,
    /// Maximum number of nodes to expand before returning a partial path.
    pub max_iterations: usize,
    /// Heuristic weight multiplier (1.0 = standard A*, >1.0 = greedy).
    pub heuristic_weight: f32,
}

impl<G: NavGraph> AStarPathfinder<G> {
    /// Creates a new A* pathfinder with the given graph.
    pub fn new(graph: G) -> Self {
        Self {
            graph,
            max_iterations: 50_000,
            heuristic_weight: 1.0,
        }
    }

    /// Sets the maximum iteration count.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Sets the heuristic weight.
    pub fn with_heuristic_weight(mut self, weight: f32) -> Self {
        self.heuristic_weight = weight;
        self
    }

    /// Run A* from `start` to `goal` on the graph.
    ///
    /// Returns `Some(Path)` on success (possibly partial if the iteration limit
    /// is hit), or `None` if start/goal are invalid or no path exists.
    pub fn find_path_on_graph(&self, start: NodeId, goal: NodeId) -> Option<Path> {
        profiling::scope!("AStarPathfinder::find_path_on_graph");

        // g_cost: best known cost from start to each node
        let mut g_costs: HashMap<NodeId, f32> = HashMap::new();
        // Parent map for path reconstruction
        let mut parents: HashMap<NodeId, NodeId> = HashMap::new();
        // Closed set
        let mut closed: HashSet<NodeId> = HashSet::new();
        // Open set (min-heap by f-cost)
        let mut open: BinaryHeap<OpenSetEntry> = BinaryHeap::new();

        let h = self.graph.heuristic(start, goal) * self.heuristic_weight;
        g_costs.insert(start, 0.0);
        open.push(OpenSetEntry {
            node_id: start,
            f_cost: h,
        });

        let mut iterations = 0usize;
        // Track the closest node to goal (for partial path fallback)
        let mut best_node = start;
        let mut best_h = h;

        while let Some(current_entry) = open.pop() {
            let current = current_entry.node_id;

            if current == goal {
                // Reconstruct path
                return Some(self.reconstruct_path(&parents, &g_costs, start, goal));
            }

            if closed.contains(&current) {
                continue;
            }
            closed.insert(current);

            iterations += 1;
            if iterations >= self.max_iterations {
                log::warn!(
                    "A* reached iteration limit ({}) without finding goal; returning partial path",
                    self.max_iterations
                );
                return Some(self.reconstruct_partial_path(&parents, &g_costs, start, best_node));
            }

            let current_g = g_costs[&current];

            for (neighbor, edge_cost) in self.graph.neighbors(current) {
                if closed.contains(&neighbor) {
                    continue;
                }

                let tentative_g = current_g + edge_cost;
                let existing_g = g_costs.get(&neighbor).copied().unwrap_or(f32::INFINITY);

                if tentative_g < existing_g {
                    g_costs.insert(neighbor, tentative_g);
                    parents.insert(neighbor, current);

                    let h =
                        self.graph.heuristic(neighbor, goal) * self.heuristic_weight;
                    let f = tentative_g + h;

                    open.push(OpenSetEntry {
                        node_id: neighbor,
                        f_cost: f,
                    });

                    // Track best node (closest to goal by heuristic)
                    if h < best_h {
                        best_h = h;
                        best_node = neighbor;
                    }
                }
            }
        }

        // Open set exhausted without reaching goal: no path exists.
        if best_node != start {
            Some(self.reconstruct_partial_path(&parents, &g_costs, start, best_node))
        } else {
            None
        }
    }

    /// Reconstruct a complete path from parent pointers.
    fn reconstruct_path(
        &self,
        parents: &HashMap<NodeId, NodeId>,
        g_costs: &HashMap<NodeId, f32>,
        start: NodeId,
        goal: NodeId,
    ) -> Path {
        let mut nodes = Vec::new();
        let mut current = goal;
        loop {
            nodes.push(current);
            if current == start {
                break;
            }
            match parents.get(&current) {
                Some(&parent) => current = parent,
                None => break,
            }
        }
        nodes.reverse();

        let cost = g_costs.get(&goal).copied().unwrap_or(0.0);
        let waypoints: Vec<Vec3> = nodes.iter().map(|&n| self.graph.node_position(n)).collect();
        Path::new(waypoints, nodes, cost)
    }

    /// Reconstruct a partial path to the best node found.
    fn reconstruct_partial_path(
        &self,
        parents: &HashMap<NodeId, NodeId>,
        g_costs: &HashMap<NodeId, f32>,
        start: NodeId,
        best: NodeId,
    ) -> Path {
        let mut nodes = Vec::new();
        let mut current = best;
        loop {
            nodes.push(current);
            if current == start {
                break;
            }
            match parents.get(&current) {
                Some(&parent) => current = parent,
                None => break,
            }
        }
        nodes.reverse();

        let cost = g_costs.get(&best).copied().unwrap_or(0.0);
        let waypoints: Vec<Vec3> = nodes.iter().map(|&n| self.graph.node_position(n)).collect();
        Path::partial(waypoints, nodes, cost)
    }
}

// ---------------------------------------------------------------------------
// GridGraph
// ---------------------------------------------------------------------------

/// 2D grid-based navigation graph with 8-directional movement.
///
/// Cells can be blocked to represent obstacles. Diagonal movement costs
/// sqrt(2) and cardinal movement costs 1.0 (scaled by `cell_size`).
#[derive(Debug, Clone)]
pub struct GridGraph {
    /// Grid width in cells.
    pub width: i32,
    /// Grid height in cells.
    pub height: i32,
    /// World-space size of each cell.
    pub cell_size: f32,
    /// Set of blocked cells.
    pub blocked: HashSet<(i32, i32)>,
    /// Per-cell cost multiplier (default 1.0). Missing entries = 1.0.
    pub cell_costs: HashMap<(i32, i32), f32>,
    /// Origin offset in world space (bottom-left corner of grid).
    pub origin: Vec3,
}

/// Diagonal cost constant.
const SQRT_2: f32 = 1.414_213_6;

/// The 8 directions for grid movement: (dx, dy, cost_multiplier).
const DIRECTIONS: [(i32, i32, f32); 8] = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, SQRT_2),
    (1, -1, SQRT_2),
    (-1, 1, SQRT_2),
    (-1, -1, SQRT_2),
];

impl GridGraph {
    /// Create a new grid graph.
    pub fn new(width: i32, height: i32, cell_size: f32) -> Self {
        Self {
            width,
            height,
            cell_size,
            blocked: HashSet::new(),
            cell_costs: HashMap::new(),
            origin: Vec3::ZERO,
        }
    }

    /// Create a grid graph with a custom origin.
    pub fn with_origin(mut self, origin: Vec3) -> Self {
        self.origin = origin;
        self
    }

    /// Set whether a cell is blocked.
    pub fn set_blocked(&mut self, x: i32, y: i32, is_blocked: bool) {
        if is_blocked {
            self.blocked.insert((x, y));
        } else {
            self.blocked.remove(&(x, y));
        }
    }

    /// Check if a cell is blocked.
    pub fn is_blocked(&self, x: i32, y: i32) -> bool {
        self.blocked.contains(&(x, y))
    }

    /// Set the traversal cost multiplier for a cell (default 1.0).
    pub fn set_cell_cost(&mut self, x: i32, y: i32, cost: f32) {
        if (cost - 1.0).abs() < 1e-6 {
            self.cell_costs.remove(&(x, y));
        } else {
            self.cell_costs.insert((x, y), cost);
        }
    }

    /// Get the traversal cost multiplier for a cell.
    pub fn cell_cost(&self, x: i32, y: i32) -> f32 {
        self.cell_costs.get(&(x, y)).copied().unwrap_or(1.0)
    }

    /// Check if a coordinate is in bounds.
    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < self.width && y < self.height
    }

    /// Convert a world-space position to grid coordinates.
    pub fn world_to_grid(&self, pos: Vec3) -> (i32, i32) {
        let local = pos - self.origin;
        let x = (local.x / self.cell_size).floor() as i32;
        let y = (local.z / self.cell_size).floor() as i32;
        (x, y)
    }

    /// Convert grid coordinates to a world-space position (center of cell).
    pub fn grid_to_world(&self, x: i32, y: i32) -> Vec3 {
        Vec3::new(
            (x as f32 + 0.5) * self.cell_size + self.origin.x,
            self.origin.y,
            (y as f32 + 0.5) * self.cell_size + self.origin.z,
        )
    }

    /// Convert grid coordinates to a NodeId.
    pub fn grid_to_node(&self, x: i32, y: i32) -> NodeId {
        NodeId::new((y * self.width + x) as u32)
    }

    /// Convert a NodeId back to grid coordinates.
    pub fn node_to_grid(&self, node: NodeId) -> (i32, i32) {
        let idx = node.0 as i32;
        (idx % self.width, idx / self.width)
    }

    /// Block a rectangular region.
    pub fn set_rect_blocked(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, is_blocked: bool) {
        let min_x = x1.min(x2);
        let max_x = x1.max(x2);
        let min_y = y1.min(y2);
        let max_y = y1.max(y2);
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                self.set_blocked(x, y, is_blocked);
            }
        }
    }

    /// Check line-of-sight between two grid cells using Bresenham's algorithm.
    /// Returns true if no blocked cell lies on the line.
    pub fn has_line_of_sight(&self, x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
        let mut x = x0;
        let mut y = y0;
        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;

        loop {
            if self.is_blocked(x, y) {
                return false;
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
        true
    }
}

impl NavGraph for GridGraph {
    fn neighbors(&self, node: NodeId) -> Vec<(NodeId, f32)> {
        let (x, y) = self.node_to_grid(node);
        let mut result = Vec::with_capacity(8);

        for &(dx, dy, base_cost) in &DIRECTIONS {
            let nx = x + dx;
            let ny = y + dy;

            if !self.in_bounds(nx, ny) || self.is_blocked(nx, ny) {
                continue;
            }

            // For diagonal movement, ensure both cardinal neighbors are passable
            // to prevent corner-cutting.
            if dx != 0 && dy != 0 {
                if self.is_blocked(x + dx, y) || self.is_blocked(x, y + dy) {
                    continue;
                }
            }

            let cost = base_cost * self.cell_size * self.cell_cost(nx, ny);
            result.push((self.grid_to_node(nx, ny), cost));
        }

        result
    }

    fn heuristic(&self, a: NodeId, b: NodeId) -> f32 {
        // Octile distance heuristic (consistent for 8-directional grids).
        let (ax, ay) = self.node_to_grid(a);
        let (bx, by) = self.node_to_grid(b);
        let dx = (bx - ax).unsigned_abs() as f32;
        let dy = (by - ay).unsigned_abs() as f32;
        let min = dx.min(dy);
        let max = dx.max(dy);
        (min * SQRT_2 + (max - min)) * self.cell_size
    }

    fn node_position(&self, node: NodeId) -> Vec3 {
        let (x, y) = self.node_to_grid(node);
        self.grid_to_world(x, y)
    }
}

// ---------------------------------------------------------------------------
// Path Smoothing
// ---------------------------------------------------------------------------

/// Smooth a grid path using line-of-sight checks to remove unnecessary waypoints.
///
/// This produces a shorter, more natural-looking path by skipping intermediate
/// nodes whenever there is a clear line of sight.
pub fn smooth_path_los(path: &Path, grid: &GridGraph) -> Path {
    if path.waypoints.len() <= 2 {
        return path.clone();
    }

    let grid_coords: Vec<(i32, i32)> = path
        .nodes
        .iter()
        .map(|&n| grid.node_to_grid(n))
        .collect();

    let mut smoothed_indices = vec![0usize];
    let mut current = 0;

    while current < grid_coords.len() - 1 {
        let mut farthest = current + 1;

        // Find the farthest reachable node with line of sight.
        for check in (current + 2)..grid_coords.len() {
            let (cx, cy) = grid_coords[current];
            let (tx, ty) = grid_coords[check];
            if grid.has_line_of_sight(cx, cy, tx, ty) {
                farthest = check;
            }
        }

        smoothed_indices.push(farthest);
        current = farthest;
    }

    let mut new_waypoints = Vec::with_capacity(smoothed_indices.len());
    let mut new_nodes = Vec::with_capacity(smoothed_indices.len());
    for &i in &smoothed_indices {
        new_waypoints.push(path.waypoints[i]);
        new_nodes.push(path.nodes[i]);
    }

    // Recompute cost
    let cost: f32 = new_waypoints
        .windows(2)
        .map(|w| (w[1] - w[0]).length())
        .sum();

    Path::new(new_waypoints, new_nodes, cost)
}

// ---------------------------------------------------------------------------
// HierarchicalPathfinder (HPA*)
// ---------------------------------------------------------------------------

/// A cluster in the hierarchical pathfinding graph.
#[derive(Debug, Clone)]
struct Cluster {
    /// Grid-space bounding box: min (x, y).
    min_x: i32,
    min_y: i32,
    /// Grid-space bounding box: max (exclusive).
    max_x: i32,
    max_y: i32,
    /// Cluster id.
    id: usize,
}

impl Cluster {
    fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.min_x && x < self.max_x && y >= self.min_y && y < self.max_y
    }
}

/// An abstract edge between two border nodes in the hierarchical graph.
#[derive(Debug, Clone)]
struct AbstractEdge {
    from: NodeId,
    to: NodeId,
    cost: f32,
}

/// A border node that sits on the boundary between two clusters.
#[derive(Debug, Clone)]
struct BorderNode {
    /// The node id in the base grid.
    grid_node: NodeId,
    /// Grid coordinates.
    x: i32,
    y: i32,
    /// The cluster this border node belongs to.
    cluster_id: usize,
    /// Abstract node id used in the hierarchical graph.
    abstract_id: NodeId,
}

/// Hierarchical pathfinder using HPA* (Hierarchical Pathfinding A*).
///
/// Divides the grid into clusters and precomputes inter-cluster edges,
/// enabling fast pathfinding over very large worlds by searching at a coarse
/// level first and refining locally.
pub struct HierarchicalPathfinder {
    /// Cluster size in grid cells.
    pub cluster_size: i32,
    /// The base grid graph.
    grid: GridGraph,
    /// Precomputed clusters.
    clusters: Vec<Cluster>,
    /// Number of clusters along each axis.
    clusters_x: i32,
    clusters_y: i32,
    /// Border nodes between clusters.
    border_nodes: Vec<BorderNode>,
    /// Abstract edges (inter-cluster and intra-cluster between border nodes).
    abstract_edges: Vec<AbstractEdge>,
    /// Adjacency list for the abstract graph: abstract_node_id -> [(neighbor_abstract_id, cost)].
    abstract_adjacency: HashMap<NodeId, Vec<(NodeId, f32)>>,
    /// Map from abstract node id to border node.
    abstract_to_border: HashMap<NodeId, usize>,
    /// Next abstract node id counter.
    next_abstract_id: u32,
    /// Maximum iterations for local A*.
    pub max_local_iterations: usize,
}

impl HierarchicalPathfinder {
    /// Creates a new hierarchical pathfinder.
    pub fn new(grid: GridGraph, cluster_size: i32) -> Self {
        Self {
            cluster_size,
            grid,
            clusters: Vec::new(),
            clusters_x: 0,
            clusters_y: 0,
            border_nodes: Vec::new(),
            abstract_edges: Vec::new(),
            abstract_adjacency: HashMap::new(),
            abstract_to_border: HashMap::new(),
            next_abstract_id: 0,
            max_local_iterations: 10_000,
        }
    }

    /// Access the underlying grid.
    pub fn grid(&self) -> &GridGraph {
        &self.grid
    }

    /// Mutably access the underlying grid.
    pub fn grid_mut(&mut self) -> &mut GridGraph {
        &mut self.grid
    }

    fn alloc_abstract_id(&mut self) -> NodeId {
        let id = NodeId::new(self.next_abstract_id);
        self.next_abstract_id += 1;
        id
    }

    /// Precompute the cluster hierarchy from the grid. Must be called before
    /// path queries.
    pub fn precompute(&mut self) {
        profiling::scope!("HierarchicalPathfinder::precompute");

        self.clusters.clear();
        self.border_nodes.clear();
        self.abstract_edges.clear();
        self.abstract_adjacency.clear();
        self.abstract_to_border.clear();
        self.next_abstract_id = 0;

        // Step 1: Create clusters.
        self.clusters_x = (self.grid.width + self.cluster_size - 1) / self.cluster_size;
        self.clusters_y = (self.grid.height + self.cluster_size - 1) / self.cluster_size;

        for cy in 0..self.clusters_y {
            for cx in 0..self.clusters_x {
                let min_x = cx * self.cluster_size;
                let min_y = cy * self.cluster_size;
                let max_x = (min_x + self.cluster_size).min(self.grid.width);
                let max_y = (min_y + self.cluster_size).min(self.grid.height);
                let id = self.clusters.len();
                self.clusters.push(Cluster {
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                    id,
                });
            }
        }

        // Step 2: Find border nodes between adjacent clusters.
        // We look at horizontal and vertical borders.
        self.find_border_nodes();

        // Step 3: Precompute intra-cluster paths between border nodes.
        self.precompute_intra_cluster_edges();

        log::info!(
            "HPA* precomputed: {} clusters, {} border nodes, {} abstract edges",
            self.clusters.len(),
            self.border_nodes.len(),
            self.abstract_edges.len()
        );
    }

    fn cluster_at(&self, x: i32, y: i32) -> Option<usize> {
        if !self.grid.in_bounds(x, y) {
            return None;
        }
        let cx = x / self.cluster_size;
        let cy = y / self.cluster_size;
        let idx = (cy * self.clusters_x + cx) as usize;
        if idx < self.clusters.len() {
            Some(idx)
        } else {
            None
        }
    }

    fn find_border_nodes(&mut self) {
        // For each pair of adjacent clusters, find entrance cells along their
        // shared edge. Consecutive passable cells are grouped into "entrances"
        // and we place border nodes at the endpoints of each entrance.

        // Horizontal borders (cluster boundary is a vertical line x = max_x of left cluster)
        for cy in 0..self.clusters_y {
            for cx in 0..(self.clusters_x - 1) {
                let left_id = (cy * self.clusters_x + cx) as usize;
                let right_id = (cy * self.clusters_x + cx + 1) as usize;
                let border_x = (cx + 1) * self.cluster_size;

                // Scan along y for passable border cells.
                let y_min = cy * self.cluster_size;
                let y_max = ((cy + 1) * self.cluster_size).min(self.grid.height);

                let mut entrance_start: Option<i32> = None;
                for y in y_min..y_max {
                    let left_passable =
                        self.grid.in_bounds(border_x - 1, y) && !self.grid.is_blocked(border_x - 1, y);
                    let right_passable =
                        self.grid.in_bounds(border_x, y) && !self.grid.is_blocked(border_x, y);

                    if left_passable && right_passable {
                        if entrance_start.is_none() {
                            entrance_start = Some(y);
                        }
                    } else {
                        if let Some(start) = entrance_start {
                            self.add_entrance_nodes(
                                border_x - 1,
                                border_x,
                                start,
                                y - 1,
                                left_id,
                                right_id,
                                true,
                            );
                            entrance_start = None;
                        }
                    }
                }
                if let Some(start) = entrance_start {
                    self.add_entrance_nodes(
                        border_x - 1,
                        border_x,
                        start,
                        y_max - 1,
                        left_id,
                        right_id,
                        true,
                    );
                }
            }
        }

        // Vertical borders (cluster boundary is a horizontal line y = max_y of top cluster)
        for cy in 0..(self.clusters_y - 1) {
            for cx in 0..self.clusters_x {
                let top_id = (cy * self.clusters_x + cx) as usize;
                let bottom_id = ((cy + 1) * self.clusters_x + cx) as usize;
                let border_y = (cy + 1) * self.cluster_size;

                let x_min = cx * self.cluster_size;
                let x_max = ((cx + 1) * self.cluster_size).min(self.grid.width);

                let mut entrance_start: Option<i32> = None;
                for x in x_min..x_max {
                    let top_passable =
                        self.grid.in_bounds(x, border_y - 1) && !self.grid.is_blocked(x, border_y - 1);
                    let bot_passable =
                        self.grid.in_bounds(x, border_y) && !self.grid.is_blocked(x, border_y);

                    if top_passable && bot_passable {
                        if entrance_start.is_none() {
                            entrance_start = Some(x);
                        }
                    } else {
                        if let Some(start) = entrance_start {
                            self.add_entrance_nodes(
                                start,
                                x - 1,
                                border_y - 1,
                                border_y,
                                top_id,
                                bottom_id,
                                false,
                            );
                            entrance_start = None;
                        }
                    }
                }
                if let Some(start) = entrance_start {
                    self.add_entrance_nodes(
                        start,
                        x_max - 1,
                        border_y - 1,
                        border_y,
                        top_id,
                        bottom_id,
                        false,
                    );
                }
            }
        }
    }

    fn add_entrance_nodes(
        &mut self,
        x1: i32,
        x2: i32,
        y1: i32,
        y2: i32,
        cluster_a: usize,
        cluster_b: usize,
        is_horizontal_border: bool,
    ) {
        // Place border nodes at the entrance. For small entrances (<=6 cells),
        // use the midpoint. For larger, use both endpoints.
        if is_horizontal_border {
            // x1 is in cluster_a, x2 is in cluster_b, y1..=y2 is the entrance span.
            let span = y2 - y1 + 1;
            if span <= 6 {
                let mid_y = (y1 + y2) / 2;
                let a_id = self.add_border_node(x1, mid_y, cluster_a);
                let b_id = self.add_border_node(x2, mid_y, cluster_b);
                self.add_abstract_edge(a_id, b_id, self.grid.cell_size);
            } else {
                let a1 = self.add_border_node(x1, y1, cluster_a);
                let b1 = self.add_border_node(x2, y1, cluster_b);
                self.add_abstract_edge(a1, b1, self.grid.cell_size);

                let a2 = self.add_border_node(x1, y2, cluster_a);
                let b2 = self.add_border_node(x2, y2, cluster_b);
                self.add_abstract_edge(a2, b2, self.grid.cell_size);
            }
        } else {
            // x1..=x2 is the entrance span, y1 is in cluster_a, y2 is in cluster_b.
            let span = x2 - x1 + 1;
            if span <= 6 {
                let mid_x = (x1 + x2) / 2;
                let a_id = self.add_border_node(mid_x, y1, cluster_a);
                let b_id = self.add_border_node(mid_x, y2, cluster_b);
                self.add_abstract_edge(a_id, b_id, self.grid.cell_size);
            } else {
                let a1 = self.add_border_node(x1, y1, cluster_a);
                let b1 = self.add_border_node(x1, y2, cluster_b);
                self.add_abstract_edge(a1, b1, self.grid.cell_size);

                let a2 = self.add_border_node(x2, y1, cluster_a);
                let b2 = self.add_border_node(x2, y2, cluster_b);
                self.add_abstract_edge(a2, b2, self.grid.cell_size);
            }
        }
    }

    fn add_border_node(&mut self, x: i32, y: i32, cluster_id: usize) -> NodeId {
        // Check if we already have a border node at this position in this cluster.
        for bn in &self.border_nodes {
            if bn.x == x && bn.y == y && bn.cluster_id == cluster_id {
                return bn.abstract_id;
            }
        }

        let abstract_id = self.alloc_abstract_id();
        let grid_node = self.grid.grid_to_node(x, y);
        self.border_nodes.push(BorderNode {
            grid_node,
            x,
            y,
            cluster_id,
            abstract_id,
        });
        let idx = self.border_nodes.len() - 1;
        self.abstract_to_border.insert(abstract_id, idx);
        self.abstract_adjacency
            .entry(abstract_id)
            .or_insert_with(Vec::new);
        abstract_id
    }

    fn add_abstract_edge(&mut self, from: NodeId, to: NodeId, cost: f32) {
        self.abstract_edges.push(AbstractEdge { from, to, cost });
        self.abstract_adjacency
            .entry(from)
            .or_insert_with(Vec::new)
            .push((to, cost));
        self.abstract_adjacency
            .entry(to)
            .or_insert_with(Vec::new)
            .push((from, cost));
    }

    fn precompute_intra_cluster_edges(&mut self) {
        // For each cluster, find all border nodes, then compute shortest paths
        // between each pair using local A* within the cluster.
        let num_clusters = self.clusters.len();
        for cluster_idx in 0..num_clusters {
            let cluster_border_ids: Vec<(NodeId, i32, i32)> = self
                .border_nodes
                .iter()
                .filter(|bn| bn.cluster_id == cluster_idx)
                .map(|bn| (bn.abstract_id, bn.x, bn.y))
                .collect();

            for i in 0..cluster_border_ids.len() {
                for j in (i + 1)..cluster_border_ids.len() {
                    let (a_id, ax, ay) = cluster_border_ids[i];
                    let (b_id, bx, by) = cluster_border_ids[j];

                    // Run local A* within the cluster bounds.
                    if let Some(cost) = self.local_astar(
                        ax,
                        ay,
                        bx,
                        by,
                        &self.clusters[cluster_idx],
                    ) {
                        self.add_abstract_edge(a_id, b_id, cost);
                    }
                }
            }
        }
    }

    /// Run A* constrained within a cluster. Returns the path cost or None.
    fn local_astar(
        &self,
        sx: i32,
        sy: i32,
        gx: i32,
        gy: i32,
        cluster: &Cluster,
    ) -> Option<f32> {
        let start = self.grid.grid_to_node(sx, sy);
        let goal = self.grid.grid_to_node(gx, gy);

        let mut g_costs: HashMap<NodeId, f32> = HashMap::new();
        let mut closed: HashSet<NodeId> = HashSet::new();
        let mut open: BinaryHeap<OpenSetEntry> = BinaryHeap::new();

        let h = self.grid.heuristic(start, goal);
        g_costs.insert(start, 0.0);
        open.push(OpenSetEntry {
            node_id: start,
            f_cost: h,
        });

        let mut iterations = 0;
        while let Some(entry) = open.pop() {
            let current = entry.node_id;
            if current == goal {
                return g_costs.get(&goal).copied();
            }
            if closed.contains(&current) {
                continue;
            }
            closed.insert(current);
            iterations += 1;
            if iterations > self.max_local_iterations {
                return None;
            }

            let current_g = g_costs[&current];
            for (neighbor, edge_cost) in self.grid.neighbors(current) {
                // Constrain to cluster bounds.
                let (nx, ny) = self.grid.node_to_grid(neighbor);
                if !cluster.contains(nx, ny) {
                    continue;
                }
                if closed.contains(&neighbor) {
                    continue;
                }
                let tentative_g = current_g + edge_cost;
                let existing_g = g_costs.get(&neighbor).copied().unwrap_or(f32::INFINITY);
                if tentative_g < existing_g {
                    g_costs.insert(neighbor, tentative_g);
                    let h = self.grid.heuristic(neighbor, goal);
                    open.push(OpenSetEntry {
                        node_id: neighbor,
                        f_cost: tentative_g + h,
                    });
                }
            }
        }
        None
    }

    /// Find a path using the hierarchical approach.
    ///
    /// 1. Insert start/goal as temporary abstract nodes.
    /// 2. Search at the abstract level.
    /// 3. Refine each abstract edge into a local path.
    pub fn find_path(&self, start: Vec3, goal: Vec3) -> Option<Path> {
        profiling::scope!("HierarchicalPathfinder::find_path");

        let (sx, sy) = self.grid.world_to_grid(start);
        let (gx, gy) = self.grid.world_to_grid(goal);

        if !self.grid.in_bounds(sx, sy) || !self.grid.in_bounds(gx, gy) {
            return None;
        }
        if self.grid.is_blocked(sx, sy) || self.grid.is_blocked(gx, gy) {
            return None;
        }

        // If start and goal are in the same cluster, just use local A*.
        let start_cluster = self.cluster_at(sx, sy)?;
        let goal_cluster = self.cluster_at(gx, gy)?;

        if start_cluster == goal_cluster {
            // Direct local search.
            return self.find_local_path(sx, sy, gx, gy, &self.clusters[start_cluster]);
        }

        // Build temporary abstract nodes for start and goal.
        let mut temp_adjacency = self.abstract_adjacency.clone();
        let start_abstract = NodeId::new(self.next_abstract_id);
        let goal_abstract = NodeId::new(self.next_abstract_id + 1);

        temp_adjacency.insert(start_abstract, Vec::new());
        temp_adjacency.insert(goal_abstract, Vec::new());

        // Connect start to all border nodes in its cluster.
        for bn in &self.border_nodes {
            if bn.cluster_id == start_cluster {
                if let Some(cost) =
                    self.local_astar(sx, sy, bn.x, bn.y, &self.clusters[start_cluster])
                {
                    temp_adjacency
                        .entry(start_abstract)
                        .or_insert_with(Vec::new)
                        .push((bn.abstract_id, cost));
                    temp_adjacency
                        .entry(bn.abstract_id)
                        .or_insert_with(Vec::new)
                        .push((start_abstract, cost));
                }
            }
        }

        // Connect goal to all border nodes in its cluster.
        for bn in &self.border_nodes {
            if bn.cluster_id == goal_cluster {
                if let Some(cost) =
                    self.local_astar(gx, gy, bn.x, bn.y, &self.clusters[goal_cluster])
                {
                    temp_adjacency
                        .entry(goal_abstract)
                        .or_insert_with(Vec::new)
                        .push((bn.abstract_id, cost));
                    temp_adjacency
                        .entry(bn.abstract_id)
                        .or_insert_with(Vec::new)
                        .push((goal_abstract, cost));
                }
            }
        }

        // A* on the abstract graph.
        let abstract_path = self.abstract_astar(
            start_abstract,
            goal_abstract,
            sx,
            sy,
            gx,
            gy,
            &temp_adjacency,
        )?;

        // Refine: convert abstract path into a concrete grid-level path.
        self.refine_path(&abstract_path, sx, sy, gx, gy)
    }

    fn abstract_astar(
        &self,
        start: NodeId,
        goal: NodeId,
        sx: i32,
        sy: i32,
        gx: i32,
        gy: i32,
        adjacency: &HashMap<NodeId, Vec<(NodeId, f32)>>,
    ) -> Option<Vec<NodeId>> {
        let mut g_costs: HashMap<NodeId, f32> = HashMap::new();
        let mut parents: HashMap<NodeId, NodeId> = HashMap::new();
        let mut closed: HashSet<NodeId> = HashSet::new();
        let mut open: BinaryHeap<OpenSetEntry> = BinaryHeap::new();

        let h = self.abstract_heuristic(start, goal, sx, sy, gx, gy);
        g_costs.insert(start, 0.0);
        open.push(OpenSetEntry {
            node_id: start,
            f_cost: h,
        });

        while let Some(entry) = open.pop() {
            let current = entry.node_id;
            if current == goal {
                let mut path = Vec::new();
                let mut c = goal;
                loop {
                    path.push(c);
                    if c == start {
                        break;
                    }
                    match parents.get(&c) {
                        Some(&p) => c = p,
                        None => break,
                    }
                }
                path.reverse();
                return Some(path);
            }
            if closed.contains(&current) {
                continue;
            }
            closed.insert(current);

            let current_g = g_costs[&current];
            if let Some(neighbors) = adjacency.get(&current) {
                for &(neighbor, edge_cost) in neighbors {
                    if closed.contains(&neighbor) {
                        continue;
                    }
                    let tentative_g = current_g + edge_cost;
                    let existing_g = g_costs.get(&neighbor).copied().unwrap_or(f32::INFINITY);
                    if tentative_g < existing_g {
                        g_costs.insert(neighbor, tentative_g);
                        parents.insert(neighbor, current);
                        let h = self.abstract_heuristic(neighbor, goal, sx, sy, gx, gy);
                        open.push(OpenSetEntry {
                            node_id: neighbor,
                            f_cost: tentative_g + h,
                        });
                    }
                }
            }
        }
        None
    }

    fn abstract_heuristic(
        &self,
        from: NodeId,
        to: NodeId,
        sx: i32,
        sy: i32,
        gx: i32,
        gy: i32,
    ) -> f32 {
        let (fx, fy) = self.abstract_node_coords(from, sx, sy);
        let (tx, ty) = self.abstract_node_coords(to, gx, gy);
        let dx = (tx - fx).unsigned_abs() as f32;
        let dy = (ty - fy).unsigned_abs() as f32;
        let min = dx.min(dy);
        let max = dx.max(dy);
        (min * SQRT_2 + (max - min)) * self.grid.cell_size
    }

    fn abstract_node_coords(&self, node: NodeId, sx: i32, sy: i32) -> (i32, i32) {
        // For temp start/goal nodes, return the actual coords.
        if node.0 == self.next_abstract_id {
            return (sx, sy);
        }
        if node.0 == self.next_abstract_id + 1 {
            // This won't happen since we don't store +1 in our coords, but handle it.
            return (sx, sy); // Will be overridden in find_path
        }
        if let Some(&idx) = self.abstract_to_border.get(&node) {
            let bn = &self.border_nodes[idx];
            (bn.x, bn.y)
        } else {
            (sx, sy)
        }
    }

    fn refine_path(
        &self,
        abstract_path: &[NodeId],
        sx: i32,
        sy: i32,
        gx: i32,
        gy: i32,
    ) -> Option<Path> {
        if abstract_path.is_empty() {
            return None;
        }

        let mut all_waypoints = Vec::new();
        let mut all_nodes = Vec::new();
        let mut total_cost = 0.0f32;

        // Convert abstract nodes to grid coordinates.
        let coords: Vec<(i32, i32)> = abstract_path
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                if i == 0 {
                    (sx, sy)
                } else if i == abstract_path.len() - 1 {
                    (gx, gy)
                } else if let Some(&idx) = self.abstract_to_border.get(&n) {
                    (self.border_nodes[idx].x, self.border_nodes[idx].y)
                } else {
                    (sx, sy) // fallback
                }
            })
            .collect();

        // For each consecutive pair, find local path.
        for i in 0..coords.len() - 1 {
            let (ax, ay) = coords[i];
            let (bx, by) = coords[i + 1];
            let cluster_a = self.cluster_at(ax, ay);
            let cluster_b = self.cluster_at(bx, by);
            // The two nodes might be in different clusters (inter-cluster edge),
            // so we do an unconstrained local A* for the segment.
            let cluster = if cluster_a == cluster_b {
                cluster_a
            } else {
                None
            };

            if let Some(local_path) = if let Some(ci) = cluster {
                self.find_local_path(ax, ay, bx, by, &self.clusters[ci])
            } else {
                // Cross-cluster: use full grid A*
                self.find_full_local_path(ax, ay, bx, by)
            } {
                let skip = if i > 0 && !all_waypoints.is_empty() { 1 } else { 0 };
                for wp in local_path.waypoints.into_iter().skip(skip) {
                    all_waypoints.push(wp);
                }
                for n in local_path.nodes.into_iter().skip(skip) {
                    all_nodes.push(n);
                }
                total_cost += local_path.cost;
            } else {
                return None;
            }
        }

        Some(Path::new(all_waypoints, all_nodes, total_cost))
    }

    fn find_local_path(
        &self,
        sx: i32,
        sy: i32,
        gx: i32,
        gy: i32,
        cluster: &Cluster,
    ) -> Option<Path> {
        let start = self.grid.grid_to_node(sx, sy);
        let goal = self.grid.grid_to_node(gx, gy);

        let mut g_costs: HashMap<NodeId, f32> = HashMap::new();
        let mut parents: HashMap<NodeId, NodeId> = HashMap::new();
        let mut closed: HashSet<NodeId> = HashSet::new();
        let mut open: BinaryHeap<OpenSetEntry> = BinaryHeap::new();

        let h = self.grid.heuristic(start, goal);
        g_costs.insert(start, 0.0);
        open.push(OpenSetEntry {
            node_id: start,
            f_cost: h,
        });

        let mut iterations = 0;
        while let Some(entry) = open.pop() {
            let current = entry.node_id;
            if current == goal {
                let cost = g_costs[&goal];
                let mut path_nodes = Vec::new();
                let mut c = goal;
                loop {
                    path_nodes.push(c);
                    if c == start {
                        break;
                    }
                    match parents.get(&c) {
                        Some(&p) => c = p,
                        None => break,
                    }
                }
                path_nodes.reverse();
                let waypoints: Vec<Vec3> =
                    path_nodes.iter().map(|&n| self.grid.node_position(n)).collect();
                return Some(Path::new(waypoints, path_nodes, cost));
            }
            if closed.contains(&current) {
                continue;
            }
            closed.insert(current);
            iterations += 1;
            if iterations > self.max_local_iterations {
                return None;
            }

            let current_g = g_costs[&current];
            for (neighbor, edge_cost) in self.grid.neighbors(current) {
                let (nx, ny) = self.grid.node_to_grid(neighbor);
                if !cluster.contains(nx, ny) {
                    continue;
                }
                if closed.contains(&neighbor) {
                    continue;
                }
                let tentative_g = current_g + edge_cost;
                let existing_g = g_costs.get(&neighbor).copied().unwrap_or(f32::INFINITY);
                if tentative_g < existing_g {
                    g_costs.insert(neighbor, tentative_g);
                    parents.insert(neighbor, current);
                    let h = self.grid.heuristic(neighbor, goal);
                    open.push(OpenSetEntry {
                        node_id: neighbor,
                        f_cost: tentative_g + h,
                    });
                }
            }
        }
        None
    }

    fn find_full_local_path(&self, sx: i32, sy: i32, gx: i32, gy: i32) -> Option<Path> {
        let pathfinder = AStarPathfinder::new(self.grid.clone())
            .with_max_iterations(self.max_local_iterations);
        let start = self.grid.grid_to_node(sx, sy);
        let goal = self.grid.grid_to_node(gx, gy);
        pathfinder.find_path_on_graph(start, goal)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_graph_basic() {
        let grid = GridGraph::new(10, 10, 1.0);
        assert!(grid.in_bounds(0, 0));
        assert!(grid.in_bounds(9, 9));
        assert!(!grid.in_bounds(10, 0));
        assert!(!grid.in_bounds(-1, 0));
    }

    #[test]
    fn test_grid_world_conversion() {
        let grid = GridGraph::new(10, 10, 2.0);
        let world = grid.grid_to_world(3, 4);
        assert!((world.x - 7.0).abs() < 0.01); // (3+0.5)*2.0 = 7.0
        assert!((world.z - 9.0).abs() < 0.01); // (4+0.5)*2.0 = 9.0

        let (gx, gy) = grid.world_to_grid(world);
        assert_eq!(gx, 3);
        assert_eq!(gy, 4);
    }

    #[test]
    fn test_grid_blocking() {
        let mut grid = GridGraph::new(5, 5, 1.0);
        assert!(!grid.is_blocked(2, 2));
        grid.set_blocked(2, 2, true);
        assert!(grid.is_blocked(2, 2));
        grid.set_blocked(2, 2, false);
        assert!(!grid.is_blocked(2, 2));
    }

    #[test]
    fn test_grid_neighbors() {
        let grid = GridGraph::new(5, 5, 1.0);
        // Center cell should have 8 neighbors.
        let node = grid.grid_to_node(2, 2);
        let neighbors = grid.neighbors(node);
        assert_eq!(neighbors.len(), 8);
    }

    #[test]
    fn test_grid_neighbors_corner() {
        let grid = GridGraph::new(5, 5, 1.0);
        let node = grid.grid_to_node(0, 0);
        let neighbors = grid.neighbors(node);
        // Corner: 3 neighbors (right, down, diagonal down-right).
        assert_eq!(neighbors.len(), 3);
    }

    #[test]
    fn test_grid_neighbors_blocked() {
        let mut grid = GridGraph::new(5, 5, 1.0);
        grid.set_blocked(3, 2, true);
        let node = grid.grid_to_node(2, 2);
        let neighbors = grid.neighbors(node);
        // One cardinal blocked removes it + its two diagonals that cut through it.
        // Blocked (3,2): removes (3,2), and diagonals (3,1) and (3,3) if they cut through.
        assert!(neighbors.iter().all(|(n, _)| {
            let (x, y) = grid.node_to_grid(*n);
            !(x == 3 && y == 2)
        }));
    }

    #[test]
    fn test_astar_straight_line() {
        let grid = GridGraph::new(10, 10, 1.0);
        let pathfinder = AStarPathfinder::new(grid);
        let start = NodeId::new(0); // (0,0)
        let goal = NodeId::new(9); // (9,0)

        let path = pathfinder.find_path_on_graph(start, goal).unwrap();
        assert!(!path.is_partial);
        assert_eq!(path.nodes.len(), 10); // 0,1,2,...,9
        assert!((path.cost - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_astar_diagonal() {
        let grid = GridGraph::new(10, 10, 1.0);
        let pathfinder = AStarPathfinder::new(grid);
        let start = NodeId::new(0); // (0,0)
        let goal = NodeId::new(10 * 9 + 9); // (9,9)

        let path = pathfinder.find_path_on_graph(start, goal).unwrap();
        assert!(!path.is_partial);
        // Pure diagonal: 10 nodes, cost = 9 * sqrt(2)
        assert_eq!(path.nodes.len(), 10);
        assert!((path.cost - 9.0 * SQRT_2).abs() < 0.1);
    }

    #[test]
    fn test_astar_obstacle_avoidance() {
        let mut grid = GridGraph::new(10, 10, 1.0);
        // Block a wall across the middle except for a gap.
        for x in 0..9 {
            grid.set_blocked(x, 5, true);
        }
        // Gap at (9, 5)

        let pathfinder = AStarPathfinder::new(grid);
        let start = NodeId::new(0); // (0,0)
        let goal = NodeId::new(10 * 9 + 0); // (0,9)

        let path = pathfinder.find_path_on_graph(start, goal).unwrap();
        assert!(!path.is_partial);
        assert!(path.nodes.len() > 10); // Must go around the wall
        // Path must not pass through any blocked cell
        let grid2 = &pathfinder.graph;
        for &node in &path.nodes {
            let (x, y) = grid2.node_to_grid(node);
            assert!(
                !grid2.is_blocked(x, y),
                "Path passes through blocked cell ({}, {})",
                x,
                y
            );
        }
    }

    #[test]
    fn test_astar_no_path() {
        let mut grid = GridGraph::new(5, 5, 1.0);
        // Completely surround (2,2)
        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx != 0 || dy != 0 {
                    grid.set_blocked(2 + dx, 2 + dy, true);
                }
            }
        }

        let pathfinder = AStarPathfinder::new(grid);
        let start = NodeId::new(0);     // (0,0)
        let goal = NodeId::new(5 * 2 + 2); // (2,2) - surrounded

        let result = pathfinder.find_path_on_graph(start, goal);
        // Goal is unreachable; should return partial or None.
        match result {
            None => {} // acceptable
            Some(path) => assert!(path.is_partial), // also acceptable
        }
    }

    #[test]
    fn test_astar_max_iterations() {
        let grid = GridGraph::new(100, 100, 1.0);
        let pathfinder = AStarPathfinder::new(grid).with_max_iterations(10);
        let start = NodeId::new(0);
        let goal = NodeId::new(100 * 99 + 99);

        let path = pathfinder.find_path_on_graph(start, goal).unwrap();
        assert!(path.is_partial);
    }

    #[test]
    fn test_path_simplify() {
        let mut path = Path::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
                Vec3::new(4.0, 0.0, 0.0),
            ],
            vec![
                NodeId::new(0),
                NodeId::new(1),
                NodeId::new(2),
                NodeId::new(3),
                NodeId::new(4),
            ],
            4.0,
        );
        path.simplify(0.1);
        // Collinear points should be removed, keeping only start and end.
        assert_eq!(path.waypoints.len(), 2);
    }

    #[test]
    fn test_line_of_sight() {
        let mut grid = GridGraph::new(10, 10, 1.0);
        assert!(grid.has_line_of_sight(0, 0, 9, 9));
        grid.set_blocked(5, 5, true);
        assert!(!grid.has_line_of_sight(0, 0, 9, 9));
        assert!(grid.has_line_of_sight(0, 0, 4, 4));
    }

    #[test]
    fn test_smooth_path() {
        let mut grid = GridGraph::new(10, 10, 1.0);
        // Block a wall.
        for y in 2..8 {
            grid.set_blocked(5, y, true);
        }

        let pathfinder = AStarPathfinder::new(grid.clone());
        let start = pathfinder.graph.grid_to_node(0, 5);
        let goal = pathfinder.graph.grid_to_node(9, 5);

        let path = pathfinder.find_path_on_graph(start, goal).unwrap();
        let smoothed = smooth_path_los(&path, &grid);
        // Smoothed path should have fewer waypoints.
        assert!(smoothed.waypoints.len() <= path.waypoints.len());
    }

    #[test]
    fn test_heuristic_octile() {
        let grid = GridGraph::new(10, 10, 1.0);
        let a = grid.grid_to_node(0, 0);
        let b = grid.grid_to_node(3, 4);
        let h = grid.heuristic(a, b);
        // Octile: min(3,4)*sqrt(2) + (4-3) = 3*1.4142 + 1 = 5.2426
        assert!((h - 5.2426).abs() < 0.01);
    }

    #[test]
    fn test_hierarchical_same_cluster() {
        let grid = GridGraph::new(20, 20, 1.0);
        let mut hpa = HierarchicalPathfinder::new(grid, 10);
        hpa.precompute();

        let start = Vec3::new(1.5, 0.0, 1.5);
        let goal = Vec3::new(5.5, 0.0, 5.5);
        let path = hpa.find_path(start, goal);
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(!p.is_partial);
        assert!(p.waypoints.len() >= 2);
    }

    #[test]
    fn test_hierarchical_cross_cluster() {
        let grid = GridGraph::new(20, 20, 1.0);
        let mut hpa = HierarchicalPathfinder::new(grid, 10);
        hpa.precompute();

        let start = Vec3::new(1.5, 0.0, 1.5);
        let goal = Vec3::new(18.5, 0.0, 18.5);
        let path = hpa.find_path(start, goal);
        assert!(path.is_some());
    }

    #[test]
    fn test_cell_cost_multiplier() {
        let mut grid = GridGraph::new(10, 10, 1.0);
        // Make a "swamp" column at x=5 that is very expensive.
        for y in 0..10 {
            grid.set_cell_cost(5, y, 10.0);
        }

        let pathfinder = AStarPathfinder::new(grid);
        let start = pathfinder.graph.grid_to_node(0, 5);
        let goal = pathfinder.graph.grid_to_node(9, 5);

        let path = pathfinder.find_path_on_graph(start, goal).unwrap();
        // Path should try to avoid the expensive column by going around.
        let crosses_x5 = path.nodes.iter().any(|n| {
            let (x, _) = pathfinder.graph.node_to_grid(*n);
            x == 5
        });
        // It may still cross if going around is even more expensive,
        // but the path should exist.
        assert!(!path.is_partial);
        let _ = crosses_x5; // just verify path is valid
    }

    #[test]
    fn test_rect_blocked() {
        let mut grid = GridGraph::new(10, 10, 1.0);
        grid.set_rect_blocked(2, 2, 4, 4, true);
        assert!(grid.is_blocked(2, 2));
        assert!(grid.is_blocked(3, 3));
        assert!(grid.is_blocked(4, 4));
        assert!(!grid.is_blocked(1, 1));
        assert!(!grid.is_blocked(5, 5));
    }
}
