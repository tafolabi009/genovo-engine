//! # Maze Generation
//!
//! Multiple algorithms for generating perfect and imperfect mazes:
//!
//! - **Recursive Backtracker (DFS)** — deep, winding passages with long corridors
//! - **Kruskal's algorithm** — uniform random spanning tree, more uniform distribution
//! - **Prim's algorithm** — tends to create shorter dead ends, more branching
//! - **Eller's algorithm** — row-by-row, memory efficient for very large/infinite mazes
//! - **Wilson's algorithm** — loop-erased random walk, uniform spanning tree
//!
//! All algorithms produce perfect mazes (exactly one path between any two cells)
//! by default. Post-processing can add loops (braid mazes) or extend to 3D.

use genovo_core::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};

// ===========================================================================
// Direction
// ===========================================================================

/// Cardinal directions for maze edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MazeDirection {
    North,
    South,
    East,
    West,
    /// For 3D mazes: up.
    Up,
    /// For 3D mazes: down.
    Down,
}

impl MazeDirection {
    /// All 2D directions.
    pub const DIRS_2D: [MazeDirection; 4] = [
        MazeDirection::North,
        MazeDirection::South,
        MazeDirection::East,
        MazeDirection::West,
    ];

    /// All 3D directions.
    pub const DIRS_3D: [MazeDirection; 6] = [
        MazeDirection::North,
        MazeDirection::South,
        MazeDirection::East,
        MazeDirection::West,
        MazeDirection::Up,
        MazeDirection::Down,
    ];

    /// Get the opposite direction.
    pub fn opposite(self) -> MazeDirection {
        match self {
            MazeDirection::North => MazeDirection::South,
            MazeDirection::South => MazeDirection::North,
            MazeDirection::East => MazeDirection::West,
            MazeDirection::West => MazeDirection::East,
            MazeDirection::Up => MazeDirection::Down,
            MazeDirection::Down => MazeDirection::Up,
        }
    }

    /// Get the (dx, dy) offset for this direction.
    pub fn offset_2d(self) -> (i32, i32) {
        match self {
            MazeDirection::North => (0, -1),
            MazeDirection::South => (0, 1),
            MazeDirection::East => (1, 0),
            MazeDirection::West => (-1, 0),
            MazeDirection::Up | MazeDirection::Down => (0, 0),
        }
    }

    /// Get the (dx, dy, dz) offset for this direction in 3D.
    pub fn offset_3d(self) -> (i32, i32, i32) {
        match self {
            MazeDirection::North => (0, -1, 0),
            MazeDirection::South => (0, 1, 0),
            MazeDirection::East => (1, 0, 0),
            MazeDirection::West => (-1, 0, 0),
            MazeDirection::Up => (0, 0, 1),
            MazeDirection::Down => (0, 0, -1),
        }
    }
}

// ===========================================================================
// Maze Cell
// ===========================================================================

/// A single cell in the maze.
///
/// Each cell tracks which walls are present. A wall is "open" (removed) if
/// there is a passage to the neighboring cell in that direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MazeCell {
    /// Whether the wall in each direction is present (true = wall exists).
    pub walls: HashMap<MazeDirection, bool>,
    /// Whether this cell has been visited during generation.
    pub visited: bool,
    /// Optional set/group ID (used by Kruskal's and Eller's).
    pub set_id: usize,
}

impl MazeCell {
    /// Create a new cell with all walls intact.
    pub fn new_2d() -> Self {
        let mut walls = HashMap::new();
        for dir in &MazeDirection::DIRS_2D {
            walls.insert(*dir, true);
        }
        Self {
            walls,
            visited: false,
            set_id: 0,
        }
    }

    /// Create a new 3D cell with all walls intact.
    pub fn new_3d() -> Self {
        let mut walls = HashMap::new();
        for dir in &MazeDirection::DIRS_3D {
            walls.insert(*dir, true);
        }
        Self {
            walls,
            visited: false,
            set_id: 0,
        }
    }

    /// Check if a wall is present in the given direction.
    pub fn has_wall(&self, dir: MazeDirection) -> bool {
        *self.walls.get(&dir).unwrap_or(&true)
    }

    /// Remove the wall in the given direction (create a passage).
    pub fn remove_wall(&mut self, dir: MazeDirection) {
        self.walls.insert(dir, false);
    }

    /// Add a wall in the given direction.
    pub fn add_wall(&mut self, dir: MazeDirection) {
        self.walls.insert(dir, true);
    }

    /// Count the number of open walls (passages).
    pub fn passage_count(&self) -> usize {
        self.walls.values().filter(|&&w| !w).count()
    }

    /// Check if this cell is a dead end (exactly 1 passage).
    pub fn is_dead_end(&self) -> bool {
        self.passage_count() == 1
    }
}

// ===========================================================================
// Maze Algorithm Selection
// ===========================================================================

/// Algorithm used for maze generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MazeAlgorithm {
    /// Recursive backtracker (DFS): deep, winding mazes with long corridors.
    RecursiveBacktracker,
    /// Kruskal's algorithm: random spanning tree, uniform distribution.
    Kruskal,
    /// Prim's algorithm: shorter dead ends, more branching near the start.
    Prim,
    /// Eller's algorithm: row-by-row, memory efficient for infinite mazes.
    Eller,
    /// Wilson's algorithm: loop-erased random walk, unbiased.
    Wilson,
}

// ===========================================================================
// Maze
// ===========================================================================

/// A maze grid.
///
/// Supports 2D mazes and multi-floor 3D mazes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Maze {
    /// Maze width.
    pub width: usize,
    /// Maze height.
    pub height: usize,
    /// Number of floors (1 for 2D).
    pub floors: usize,
    /// Flat array of cells. Indexed as `floor * (width * height) + y * width + x`.
    pub cells: Vec<MazeCell>,
    /// The algorithm used to generate this maze.
    pub algorithm: MazeAlgorithm,
}

impl Maze {
    /// Create a new empty 2D maze (all walls intact).
    pub fn new_2d(width: usize, height: usize) -> Self {
        let cells = (0..width * height).map(|_| MazeCell::new_2d()).collect();
        Self {
            width,
            height,
            floors: 1,
            cells,
            algorithm: MazeAlgorithm::RecursiveBacktracker,
        }
    }

    /// Create a new empty 3D maze.
    pub fn new_3d(width: usize, height: usize, floors: usize) -> Self {
        let cells = (0..width * height * floors)
            .map(|_| MazeCell::new_3d())
            .collect();
        Self {
            width,
            height,
            floors,
            cells,
            algorithm: MazeAlgorithm::RecursiveBacktracker,
        }
    }

    /// Convert (x, y, floor) to a flat index.
    fn index(&self, x: usize, y: usize, floor: usize) -> usize {
        floor * (self.width * self.height) + y * self.width + x
    }

    /// Check if coordinates are in bounds.
    fn in_bounds(&self, x: i32, y: i32, floor: i32) -> bool {
        x >= 0
            && y >= 0
            && floor >= 0
            && (x as usize) < self.width
            && (y as usize) < self.height
            && (floor as usize) < self.floors
    }

    /// Get a reference to the cell at (x, y, floor).
    pub fn cell(&self, x: usize, y: usize, floor: usize) -> &MazeCell {
        &self.cells[self.index(x, y, floor)]
    }

    /// Get a mutable reference to the cell at (x, y, floor).
    pub fn cell_mut(&mut self, x: usize, y: usize, floor: usize) -> &mut MazeCell {
        let idx = self.index(x, y, floor);
        &mut self.cells[idx]
    }

    /// Remove the wall between two adjacent cells.
    pub fn remove_wall_between(
        &mut self,
        x1: usize,
        y1: usize,
        f1: usize,
        x2: usize,
        y2: usize,
        f2: usize,
    ) {
        // Determine direction from (x1,y1,f1) to (x2,y2,f2).
        let dx = x2 as i32 - x1 as i32;
        let dy = y2 as i32 - y1 as i32;
        let df = f2 as i32 - f1 as i32;

        let dir = if dx == 1 {
            MazeDirection::East
        } else if dx == -1 {
            MazeDirection::West
        } else if dy == 1 {
            MazeDirection::South
        } else if dy == -1 {
            MazeDirection::North
        } else if df == 1 {
            MazeDirection::Up
        } else if df == -1 {
            MazeDirection::Down
        } else {
            return;
        };

        self.cell_mut(x1, y1, f1).remove_wall(dir);
        self.cell_mut(x2, y2, f2).remove_wall(dir.opposite());
    }

    /// Get the 2D neighbors of a cell.
    #[allow(dead_code)]
    fn neighbors_2d(&self, x: usize, y: usize) -> Vec<(usize, usize, usize, MazeDirection)> {
        let mut result = Vec::new();
        for &dir in &MazeDirection::DIRS_2D {
            let (dx, dy) = dir.offset_2d();
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if self.in_bounds(nx, ny, 0) {
                result.push((nx as usize, ny as usize, 0, dir));
            }
        }
        result
    }

    /// Get the 3D neighbors of a cell.
    #[allow(dead_code)]
    fn neighbors_3d(&self, x: usize, y: usize, floor: usize) -> Vec<(usize, usize, usize, MazeDirection)> {
        let mut result = Vec::new();
        for &dir in &MazeDirection::DIRS_3D {
            let (dx, dy, dz) = dir.offset_3d();
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nf = floor as i32 + dz;
            if self.in_bounds(nx, ny, nf) {
                result.push((nx as usize, ny as usize, nf as usize, dir));
            }
        }
        result
    }

    /// Render the maze as ASCII art (2D, first floor only).
    pub fn to_ascii(&self) -> String {
        let mut result = String::new();

        // Top border.
        result.push('+');
        for x in 0..self.width {
            let cell = self.cell(x, 0, 0);
            if cell.has_wall(MazeDirection::North) {
                result.push_str("---+");
            } else {
                result.push_str("   +");
            }
        }
        result.push('\n');

        for y in 0..self.height {
            // Cell row.
            result.push('|');
            for x in 0..self.width {
                result.push_str("   ");
                let cell = self.cell(x, y, 0);
                if x < self.width - 1 && !cell.has_wall(MazeDirection::East) {
                    result.push(' ');
                } else {
                    result.push('|');
                }
            }
            result.push('\n');

            // Bottom wall row.
            result.push('+');
            for x in 0..self.width {
                let cell = self.cell(x, y, 0);
                if y < self.height - 1 && !cell.has_wall(MazeDirection::South) {
                    result.push_str("   +");
                } else {
                    result.push_str("---+");
                }
            }
            result.push('\n');
        }

        result
    }

    /// Solve the maze using A* from start to end.
    ///
    /// Returns the path as a list of (x, y) coordinates, or `None` if no
    /// path exists.
    pub fn solve(
        &self,
        start: (usize, usize),
        end: (usize, usize),
    ) -> Option<Vec<(usize, usize)>> {
        self.solve_3d((start.0, start.1, 0), (end.0, end.1, 0))
            .map(|path| path.into_iter().map(|(x, y, _)| (x, y)).collect())
    }

    /// Solve the 3D maze using A* from start to end.
    pub fn solve_3d(
        &self,
        start: (usize, usize, usize),
        end: (usize, usize, usize),
    ) -> Option<Vec<(usize, usize, usize)>> {
        let heuristic = |a: (usize, usize, usize), b: (usize, usize, usize)| -> f32 {
            let dx = (a.0 as f32 - b.0 as f32).abs();
            let dy = (a.1 as f32 - b.1 as f32).abs();
            let dz = (a.2 as f32 - b.2 as f32).abs();
            dx + dy + dz
        };

        #[derive(Debug, PartialEq)]
        struct Node {
            pos: (usize, usize, usize),
            f: f32,
        }
        impl Eq for Node {}
        impl PartialOrd for Node {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Node {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other.f.partial_cmp(&self.f).unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        let mut open = BinaryHeap::new();
        let mut came_from: HashMap<(usize, usize, usize), (usize, usize, usize)> = HashMap::new();
        let mut g_score: HashMap<(usize, usize, usize), f32> = HashMap::new();

        g_score.insert(start, 0.0);
        open.push(Node {
            pos: start,
            f: heuristic(start, end),
        });

        while let Some(current) = open.pop() {
            if current.pos == end {
                let mut path = vec![end];
                let mut pos = end;
                while let Some(&prev) = came_from.get(&pos) {
                    path.push(prev);
                    pos = prev;
                }
                path.reverse();
                return Some(path);
            }

            let current_g = *g_score.get(&current.pos).unwrap_or(&f32::MAX);
            let cell = self.cell(current.pos.0, current.pos.1, current.pos.2);

            let dirs = if self.floors > 1 {
                &MazeDirection::DIRS_3D[..]
            } else {
                &MazeDirection::DIRS_2D[..]
            };

            for &dir in dirs {
                if cell.has_wall(dir) {
                    continue; // Wall blocks passage.
                }

                let (dx, dy, dz) = dir.offset_3d();
                let nx = current.pos.0 as i32 + dx;
                let ny = current.pos.1 as i32 + dy;
                let nf = current.pos.2 as i32 + dz;

                if !self.in_bounds(nx, ny, nf) {
                    continue;
                }

                let neighbor = (nx as usize, ny as usize, nf as usize);
                let tentative_g = current_g + 1.0;
                let best_g = *g_score.get(&neighbor).unwrap_or(&f32::MAX);

                if tentative_g < best_g {
                    came_from.insert(neighbor, current.pos);
                    g_score.insert(neighbor, tentative_g);
                    open.push(Node {
                        pos: neighbor,
                        f: tentative_g + heuristic(neighbor, end),
                    });
                }
            }
        }

        None
    }

    /// Count the number of dead ends in the maze.
    pub fn dead_end_count(&self) -> usize {
        self.cells.iter().filter(|c| c.is_dead_end()).count()
    }

    /// Convert dead ends into loops (braid the maze).
    ///
    /// `removal_chance` controls how many dead ends to remove (0.0 = none,
    /// 1.0 = all).
    pub fn braid(&mut self, removal_chance: f32, rng: &mut Rng) {
        for floor in 0..self.floors {
            for y in 0..self.height {
                for x in 0..self.width {
                    if !self.cell(x, y, floor).is_dead_end() {
                        continue;
                    }
                    if rng.next_f32() > removal_chance {
                        continue;
                    }

                    // Find a wall to remove.
                    let dirs = if self.floors > 1 {
                        MazeDirection::DIRS_3D.to_vec()
                    } else {
                        MazeDirection::DIRS_2D.to_vec()
                    };

                    let mut wall_dirs: Vec<MazeDirection> = dirs
                        .into_iter()
                        .filter(|&d| self.cell(x, y, floor).has_wall(d))
                        .collect();

                    rng.shuffle(&mut wall_dirs);

                    for dir in wall_dirs {
                        let (dx, dy, dz) = dir.offset_3d();
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nf = floor as i32 + dz;

                        if self.in_bounds(nx, ny, nf) {
                            self.remove_wall_between(
                                x,
                                y,
                                floor,
                                nx as usize,
                                ny as usize,
                                nf as usize,
                            );
                            break;
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Algorithm: Recursive Backtracker (DFS)
// ===========================================================================

/// Generate a maze using the recursive backtracker (DFS) algorithm.
///
/// Produces deep, winding mazes with long corridors. Tends to have a longer
/// average path between any two points.
pub fn generate_recursive_backtracker(
    width: usize,
    height: usize,
    seed: u64,
) -> Maze {
    let mut maze = Maze::new_2d(width, height);
    maze.algorithm = MazeAlgorithm::RecursiveBacktracker;
    let mut rng = Rng::new(seed);

    let mut stack: Vec<(usize, usize)> = Vec::new();
    let start_x = rng.range_i32(0, width as i32) as usize;
    let start_y = rng.range_i32(0, height as i32) as usize;

    maze.cell_mut(start_x, start_y, 0).visited = true;
    stack.push((start_x, start_y));

    while let Some(&(x, y)) = stack.last() {
        // Find unvisited neighbors.
        let mut neighbors: Vec<(usize, usize, MazeDirection)> = Vec::new();
        for &dir in &MazeDirection::DIRS_2D {
            let (dx, dy) = dir.offset_2d();
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if maze.in_bounds(nx, ny, 0) {
                let ux = nx as usize;
                let uy = ny as usize;
                if !maze.cell(ux, uy, 0).visited {
                    neighbors.push((ux, uy, dir));
                }
            }
        }

        if neighbors.is_empty() {
            stack.pop();
        } else {
            // Choose a random unvisited neighbor.
            let idx = rng.range_i32(0, neighbors.len() as i32) as usize;
            let (nx, ny, dir) = neighbors[idx];

            // Remove walls between current and chosen.
            maze.cell_mut(x, y, 0).remove_wall(dir);
            maze.cell_mut(nx, ny, 0).remove_wall(dir.opposite());
            maze.cell_mut(nx, ny, 0).visited = true;

            stack.push((nx, ny));
        }
    }

    maze
}

// ===========================================================================
// Algorithm: Kruskal's
// ===========================================================================

/// Generate a maze using Kruskal's algorithm (random spanning tree).
///
/// Produces mazes with a more uniform distribution of passage directions.
pub fn generate_kruskal(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new_2d(width, height);
    maze.algorithm = MazeAlgorithm::Kruskal;
    let mut rng = Rng::new(seed);

    // Union-Find data structure.
    let total = width * height;
    let mut parent: Vec<usize> = (0..total).collect();
    let mut rank: Vec<usize> = vec![0; total];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], a: usize, b: usize) -> bool {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return false;
        }
        if rank[ra] < rank[rb] {
            parent[ra] = rb;
        } else if rank[ra] > rank[rb] {
            parent[rb] = ra;
        } else {
            parent[rb] = ra;
            rank[ra] += 1;
        }
        true
    }

    // Build list of all possible edges.
    let mut edges: Vec<(usize, usize, usize, usize, MazeDirection)> = Vec::new();

    for y in 0..height {
        for x in 0..width {
            if x + 1 < width {
                edges.push((x, y, x + 1, y, MazeDirection::East));
            }
            if y + 1 < height {
                edges.push((x, y, x, y + 1, MazeDirection::South));
            }
        }
    }

    // Shuffle edges randomly.
    rng.shuffle(&mut edges);

    // Process edges in random order.
    for (x1, y1, x2, y2, dir) in edges {
        let idx1 = y1 * width + x1;
        let idx2 = y2 * width + x2;

        if union(&mut parent, &mut rank, idx1, idx2) {
            maze.cell_mut(x1, y1, 0).remove_wall(dir);
            maze.cell_mut(x2, y2, 0).remove_wall(dir.opposite());
        }
    }

    maze
}

// ===========================================================================
// Algorithm: Prim's
// ===========================================================================

/// Generate a maze using Prim's algorithm.
///
/// Tends to create shorter dead ends and more branching near the starting
/// point. The maze grows outward like a tree.
pub fn generate_prim(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new_2d(width, height);
    maze.algorithm = MazeAlgorithm::Prim;
    let mut rng = Rng::new(seed);

    // Use a weighted frontier (random priority).
    #[derive(Debug, PartialEq)]
    struct FrontierEdge {
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
        dir: MazeDirection,
        priority: u32,
    }
    impl Eq for FrontierEdge {}
    impl PartialOrd for FrontierEdge {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for FrontierEdge {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            // Higher priority first (max-heap).
            self.priority.cmp(&other.priority)
        }
    }

    let start_x = rng.range_i32(0, width as i32) as usize;
    let start_y = rng.range_i32(0, height as i32) as usize;
    maze.cell_mut(start_x, start_y, 0).visited = true;

    let mut frontier: BinaryHeap<FrontierEdge> = BinaryHeap::new();

    // Add initial frontier edges.
    let add_frontier = |x: usize, y: usize, frontier: &mut BinaryHeap<FrontierEdge>, maze: &Maze, rng: &mut Rng| {
        for &dir in &MazeDirection::DIRS_2D {
            let (dx, dy) = dir.offset_2d();
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if maze.in_bounds(nx, ny, 0) {
                let ux = nx as usize;
                let uy = ny as usize;
                if !maze.cell(ux, uy, 0).visited {
                    frontier.push(FrontierEdge {
                        x1: x,
                        y1: y,
                        x2: ux,
                        y2: uy,
                        dir,
                        priority: rng.next_u32(),
                    });
                }
            }
        }
    };

    add_frontier(start_x, start_y, &mut frontier, &maze, &mut rng);

    while let Some(edge) = frontier.pop() {
        if maze.cell(edge.x2, edge.y2, 0).visited {
            continue;
        }

        // Connect the cells.
        maze.cell_mut(edge.x1, edge.y1, 0).remove_wall(edge.dir);
        maze.cell_mut(edge.x2, edge.y2, 0).remove_wall(edge.dir.opposite());
        maze.cell_mut(edge.x2, edge.y2, 0).visited = true;

        add_frontier(edge.x2, edge.y2, &mut frontier, &maze, &mut rng);
    }

    maze
}

// ===========================================================================
// Algorithm: Eller's
// ===========================================================================

/// Generate a maze using Eller's algorithm.
///
/// Processes the maze row by row, making it memory efficient and suitable
/// for generating very large or infinite mazes. At each row, cells are
/// grouped into sets; the algorithm merges sets horizontally and extends
/// sets vertically while maintaining the perfect maze property.
pub fn generate_eller(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new_2d(width, height);
    maze.algorithm = MazeAlgorithm::Eller;
    let mut rng = Rng::new(seed);

    // Current row's set assignments.
    let mut row_sets: Vec<usize> = (0..width).collect();
    let mut next_set_id = width;

    for y in 0..height {
        // Step 1: Randomly join adjacent cells in the same row.
        for x in 0..width.saturating_sub(1) {
            let is_last_row = y == height - 1;

            // If not in the same set, maybe join them.
            if row_sets[x] != row_sets[x + 1] {
                let should_join = if is_last_row {
                    true // Must join on last row to ensure connectivity.
                } else {
                    rng.bool(0.5)
                };

                if should_join {
                    // Merge sets.
                    let old_set = row_sets[x + 1];
                    let new_set = row_sets[x];
                    for s in row_sets.iter_mut() {
                        if *s == old_set {
                            *s = new_set;
                        }
                    }

                    // Remove wall.
                    maze.cell_mut(x, y, 0).remove_wall(MazeDirection::East);
                    maze.cell_mut(x + 1, y, 0).remove_wall(MazeDirection::West);
                }
            }
        }

        if y == height - 1 {
            break; // Don't extend downward from the last row.
        }

        // Step 2: For each set, ensure at least one cell extends downward.
        // Group cells by set.
        let mut set_cells: HashMap<usize, Vec<usize>> = HashMap::new();
        for x in 0..width {
            set_cells.entry(row_sets[x]).or_default().push(x);
        }

        let mut new_row_sets = vec![0usize; width];
        let mut extended: HashSet<usize> = HashSet::new();

        for (set_id, cells) in &set_cells {
            // Ensure at least one cell from this set extends downward.
            let mut at_least_one = false;

            for &x in cells {
                let extend_down = if !at_least_one && x == *cells.last().unwrap() {
                    true // Force the last cell to extend if none have.
                } else {
                    rng.bool(0.5)
                };

                if extend_down {
                    // Remove wall between (x, y) and (x, y+1).
                    maze.cell_mut(x, y, 0).remove_wall(MazeDirection::South);
                    maze.cell_mut(x, y + 1, 0).remove_wall(MazeDirection::North);
                    new_row_sets[x] = *set_id;
                    extended.insert(x);
                    at_least_one = true;
                }
            }
        }

        // Assign new set IDs to cells that didn't extend.
        for x in 0..width {
            if !extended.contains(&x) {
                new_row_sets[x] = next_set_id;
                next_set_id += 1;
            }
        }

        row_sets = new_row_sets;
    }

    maze
}

// ===========================================================================
// Algorithm: Wilson's
// ===========================================================================

/// Generate a maze using Wilson's algorithm (loop-erased random walk).
///
/// Produces a uniform spanning tree — every possible spanning tree has
/// equal probability of being generated. The algorithm starts with one
/// cell in the maze and then performs random walks from unvisited cells,
/// erasing any loops, until the walk reaches a visited cell.
pub fn generate_wilson(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new_2d(width, height);
    maze.algorithm = MazeAlgorithm::Wilson;
    let mut rng = Rng::new(seed);
    let total = width * height;

    let mut in_maze = vec![false; total];

    // Start with one random cell in the maze.
    let start = rng.range_i32(0, total as i32) as usize;
    in_maze[start] = true;
    maze.cell_mut(start % width, start / width, 0).visited = true;

    let mut remaining: Vec<usize> = (0..total).filter(|&i| i != start).collect();

    while !remaining.is_empty() {
        // Pick a random unvisited cell.
        let start_idx = rng.range_i32(0, remaining.len() as i32) as usize;
        let walk_start = remaining[start_idx];

        // Perform loop-erased random walk.
        let mut walk: Vec<usize> = vec![walk_start];
        let mut walk_dirs: Vec<MazeDirection> = Vec::new();
        let mut walk_set: HashMap<usize, usize> = HashMap::new();
        walk_set.insert(walk_start, 0);

        loop {
            let current = *walk.last().unwrap();
            let cx = current % width;
            let cy = current / width;

            // Pick a random neighbor.
            let mut neighbors: Vec<(usize, MazeDirection)> = Vec::new();
            for &dir in &MazeDirection::DIRS_2D {
                let (dx, dy) = dir.offset_2d();
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if maze.in_bounds(nx, ny, 0) {
                    let nidx = ny as usize * width + nx as usize;
                    neighbors.push((nidx, dir));
                }
            }

            let chosen_idx = rng.range_i32(0, neighbors.len() as i32) as usize;
            let (next, dir) = neighbors[chosen_idx];

            if in_maze[next] {
                // Reached the maze — add the walk path.
                walk_dirs.push(dir);
                walk.push(next);

                // Carve passages along the walk.
                for i in 0..walk.len() - 1 {
                    let c1 = walk[i];
                    let c2 = walk[i + 1];
                    let c1x = c1 % width;
                    let c1y = c1 / width;
                    let c2x = c2 % width;
                    let c2y = c2 / width;

                    let d = walk_dirs[i];
                    maze.cell_mut(c1x, c1y, 0).remove_wall(d);
                    maze.cell_mut(c2x, c2y, 0).remove_wall(d.opposite());
                    in_maze[c1] = true;
                    maze.cell_mut(c1x, c1y, 0).visited = true;
                }

                break;
            } else if let Some(&loop_idx) = walk_set.get(&next) {
                // Loop detected — erase the loop by truncating back to the
                // first occurrence of `next`. Keep the cell at `loop_idx`
                // (which is `next`), and discard everything after it.
                walk.truncate(loop_idx + 1);
                walk_dirs.truncate(loop_idx);
                // Remove erased cells from the walk set.
                walk_set.retain(|_, v| *v <= loop_idx);
            } else {
                // No loop — continue the walk.
                let walk_pos = walk.len();
                walk_set.insert(next, walk_pos);
                walk_dirs.push(dir);
                walk.push(next);
            }
        }

        // Remove visited cells from remaining.
        remaining.retain(|&idx| !in_maze[idx]);
    }

    maze
}

// ===========================================================================
// 3D Maze Generation
// ===========================================================================

/// Generate a 3D maze using recursive backtracker.
pub fn generate_3d_maze(
    width: usize,
    height: usize,
    floors: usize,
    seed: u64,
) -> Maze {
    let mut maze = Maze::new_3d(width, height, floors);
    let mut rng = Rng::new(seed);

    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    let start_x = rng.range_i32(0, width as i32) as usize;
    let start_y = rng.range_i32(0, height as i32) as usize;
    let start_f = 0usize;

    maze.cell_mut(start_x, start_y, start_f).visited = true;
    stack.push((start_x, start_y, start_f));

    while let Some(&(x, y, f)) = stack.last() {
        let mut neighbors: Vec<(usize, usize, usize, MazeDirection)> = Vec::new();

        for &dir in &MazeDirection::DIRS_3D {
            let (dx, dy, dz) = dir.offset_3d();
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nf = f as i32 + dz;
            if maze.in_bounds(nx, ny, nf) {
                let ux = nx as usize;
                let uy = ny as usize;
                let uf = nf as usize;
                if !maze.cell(ux, uy, uf).visited {
                    neighbors.push((ux, uy, uf, dir));
                }
            }
        }

        if neighbors.is_empty() {
            stack.pop();
        } else {
            let idx = rng.range_i32(0, neighbors.len() as i32) as usize;
            let (nx, ny, nf, dir) = neighbors[idx];

            maze.cell_mut(x, y, f).remove_wall(dir);
            maze.cell_mut(nx, ny, nf).remove_wall(dir.opposite());
            maze.cell_mut(nx, ny, nf).visited = true;

            stack.push((nx, ny, nf));
        }
    }

    maze
}

// ===========================================================================
// Convenience: generate with algorithm choice
// ===========================================================================

/// Generate a 2D maze using the specified algorithm.
pub fn generate_maze(
    algorithm: MazeAlgorithm,
    width: usize,
    height: usize,
    seed: u64,
) -> Maze {
    match algorithm {
        MazeAlgorithm::RecursiveBacktracker => generate_recursive_backtracker(width, height, seed),
        MazeAlgorithm::Kruskal => generate_kruskal(width, height, seed),
        MazeAlgorithm::Prim => generate_prim(width, height, seed),
        MazeAlgorithm::Eller => generate_eller(width, height, seed),
        MazeAlgorithm::Wilson => generate_wilson(width, height, seed),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_perfect_maze(maze: &Maze) {
        // A perfect maze with W*H cells should have exactly W*H - 1 passages
        // (it's a spanning tree).
        let total_cells = maze.width * maze.height * maze.floors;
        let mut passage_count = 0;

        for f in 0..maze.floors {
            for y in 0..maze.height {
                for x in 0..maze.width {
                    let cell = maze.cell(x, y, f);
                    // Count passages going east and south (to avoid double counting).
                    if !cell.has_wall(MazeDirection::East) {
                        passage_count += 1;
                    }
                    if !cell.has_wall(MazeDirection::South) {
                        passage_count += 1;
                    }
                    if maze.floors > 1 && !cell.has_wall(MazeDirection::Up) {
                        passage_count += 1;
                    }
                }
            }
        }

        assert_eq!(
            passage_count,
            total_cells - 1,
            "Perfect maze should have exactly {} passages, got {}",
            total_cells - 1,
            passage_count
        );
    }

    fn assert_solvable(maze: &Maze) {
        let solution = maze.solve((0, 0), (maze.width - 1, maze.height - 1));
        assert!(solution.is_some(), "Maze should be solvable");
        let path = solution.unwrap();
        assert_eq!(path[0], (0, 0), "Path should start at (0,0)");
        assert_eq!(
            *path.last().unwrap(),
            (maze.width - 1, maze.height - 1),
            "Path should end at bottom-right"
        );
    }

    #[test]
    fn test_recursive_backtracker() {
        let maze = generate_recursive_backtracker(10, 10, 42);
        assert_eq!(maze.algorithm, MazeAlgorithm::RecursiveBacktracker);
        assert_perfect_maze(&maze);
        assert_solvable(&maze);
    }

    #[test]
    fn test_kruskal() {
        let maze = generate_kruskal(10, 10, 42);
        assert_eq!(maze.algorithm, MazeAlgorithm::Kruskal);
        assert_perfect_maze(&maze);
        assert_solvable(&maze);
    }

    #[test]
    fn test_prim() {
        let maze = generate_prim(10, 10, 42);
        assert_eq!(maze.algorithm, MazeAlgorithm::Prim);
        assert_perfect_maze(&maze);
        assert_solvable(&maze);
    }

    #[test]
    fn test_eller() {
        let maze = generate_eller(10, 10, 42);
        assert_eq!(maze.algorithm, MazeAlgorithm::Eller);
        // Eller's produces a perfect maze.
        assert_perfect_maze(&maze);
        assert_solvable(&maze);
    }

    #[test]
    fn test_wilson() {
        let maze = generate_wilson(8, 8, 42);
        assert_eq!(maze.algorithm, MazeAlgorithm::Wilson);
        assert_perfect_maze(&maze);
        assert_solvable(&maze);
    }

    #[test]
    fn test_deterministic() {
        let m1 = generate_recursive_backtracker(8, 8, 42);
        let m2 = generate_recursive_backtracker(8, 8, 42);

        for y in 0..8 {
            for x in 0..8 {
                let c1 = m1.cell(x, y, 0);
                let c2 = m2.cell(x, y, 0);
                for &dir in &MazeDirection::DIRS_2D {
                    assert_eq!(
                        c1.has_wall(dir),
                        c2.has_wall(dir),
                        "Mazes with same seed should be identical at ({x},{y}) {dir:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_different_seeds() {
        let m1 = generate_recursive_backtracker(8, 8, 1);
        let m2 = generate_recursive_backtracker(8, 8, 2);

        let mut differ = false;
        for y in 0..8 {
            for x in 0..8 {
                for &dir in &MazeDirection::DIRS_2D {
                    if m1.cell(x, y, 0).has_wall(dir) != m2.cell(x, y, 0).has_wall(dir) {
                        differ = true;
                    }
                }
            }
        }
        assert!(differ, "Different seeds should produce different mazes");
    }

    #[test]
    fn test_ascii_output() {
        let maze = generate_recursive_backtracker(3, 3, 42);
        let ascii = maze.to_ascii();
        assert!(!ascii.is_empty());
        assert!(ascii.contains('+'));
        assert!(ascii.contains('|'));
    }

    #[test]
    fn test_dead_ends() {
        let maze = generate_recursive_backtracker(10, 10, 42);
        let dead_ends = maze.dead_end_count();
        assert!(dead_ends > 0, "Maze should have dead ends");
    }

    #[test]
    fn test_braid_reduces_dead_ends() {
        let mut maze = generate_recursive_backtracker(10, 10, 42);
        let initial_dead_ends = maze.dead_end_count();

        let mut rng = Rng::new(42);
        maze.braid(1.0, &mut rng); // Remove all dead ends.

        let final_dead_ends = maze.dead_end_count();
        assert!(
            final_dead_ends < initial_dead_ends,
            "Braiding should reduce dead ends: {initial_dead_ends} -> {final_dead_ends}"
        );
    }

    #[test]
    fn test_3d_maze() {
        let maze = generate_3d_maze(5, 5, 3, 42);
        assert_eq!(maze.floors, 3);
        assert_perfect_maze(&maze);

        // Should be solvable across floors.
        let solution = maze.solve_3d((0, 0, 0), (4, 4, 2));
        assert!(solution.is_some(), "3D maze should be solvable across floors");
    }

    #[test]
    fn test_generate_maze_dispatch() {
        // Test the convenience function dispatches correctly.
        let algorithms = [
            MazeAlgorithm::RecursiveBacktracker,
            MazeAlgorithm::Kruskal,
            MazeAlgorithm::Prim,
            MazeAlgorithm::Eller,
            MazeAlgorithm::Wilson,
        ];

        for &algo in &algorithms {
            let maze = generate_maze(algo, 6, 6, 42);
            assert_eq!(maze.algorithm, algo);
            assert_solvable(&maze);
        }
    }

    #[test]
    fn test_cell_properties() {
        let cell = MazeCell::new_2d();
        assert!(cell.has_wall(MazeDirection::North));
        assert!(cell.has_wall(MazeDirection::East));
        assert_eq!(cell.passage_count(), 0);
        assert!(!cell.is_dead_end());
    }

    #[test]
    fn test_cell_remove_wall() {
        let mut cell = MazeCell::new_2d();
        cell.remove_wall(MazeDirection::North);
        assert!(!cell.has_wall(MazeDirection::North));
        assert!(cell.has_wall(MazeDirection::South));
        assert_eq!(cell.passage_count(), 1);
        assert!(cell.is_dead_end());
    }

    #[test]
    fn test_direction_opposite() {
        assert_eq!(MazeDirection::North.opposite(), MazeDirection::South);
        assert_eq!(MazeDirection::East.opposite(), MazeDirection::West);
        assert_eq!(MazeDirection::Up.opposite(), MazeDirection::Down);
    }

    #[test]
    fn test_large_maze() {
        // Test that generation works for larger mazes without panicking.
        let maze = generate_recursive_backtracker(50, 50, 42);
        assert_eq!(maze.width, 50);
        assert_eq!(maze.height, 50);
        assert_solvable(&maze);
    }

    #[test]
    fn test_solve_path_validity() {
        let maze = generate_recursive_backtracker(8, 8, 42);
        let path = maze.solve((0, 0), (7, 7)).unwrap();

        // Check that each step is to an adjacent cell with no wall between.
        for i in 0..path.len() - 1 {
            let (x1, y1) = path[i];
            let (x2, y2) = path[i + 1];

            let dx = x2 as i32 - x1 as i32;
            let dy = y2 as i32 - y1 as i32;

            // Should be exactly one step in a cardinal direction.
            assert!(
                (dx.abs() + dy.abs()) == 1,
                "Path step should be to adjacent cell: ({x1},{y1}) -> ({x2},{y2})"
            );

            // Check no wall between the two cells.
            let dir = if dx == 1 {
                MazeDirection::East
            } else if dx == -1 {
                MazeDirection::West
            } else if dy == 1 {
                MazeDirection::South
            } else {
                MazeDirection::North
            };

            assert!(
                !maze.cell(x1, y1, 0).has_wall(dir),
                "Path should not cross walls at ({x1},{y1}) going {dir:?}"
            );
        }
    }
}
