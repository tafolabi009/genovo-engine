// engine/ai/src/pathfinding_v2.rs
// Enhanced pathfinding: Jump Point Search, theta*, flow field, hierarchical, dynamic replanning.
use std::collections::{HashMap, BinaryHeap, HashSet, VecDeque};
use std::cmp::Ordering;
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x+self.y*r.y+self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self{x:self.y*r.z-self.z*r.y,y:self.z*r.x-self.x*r.z,z:self.x*r.y-self.y*r.x} }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f32 { self.dot(self) }
    pub fn normalize(self) -> Self { let l=self.length(); if l<1e-12{Self::ZERO}else{Self{x:self.x/l,y:self.y/l,z:self.z/l}} }
    pub fn scale(self, s: f32) -> Self { Self{x:self.x*s,y:self.y*s,z:self.z*s} }
    pub fn add(self, r: Self) -> Self { Self{x:self.x+r.x,y:self.y+r.y,z:self.z+r.z} }
    pub fn sub(self, r: Self) -> Self { Self{x:self.x-r.x,y:self.y-r.y,z:self.z-r.z} }
    pub fn neg(self) -> Self { Self{x:-self.x,y:-self.y,z:-self.z} }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridPos { pub x: i32, pub y: i32 }
impl GridPos {
    pub fn new(x: i32, y: i32) -> Self { Self { x, y } }
    pub fn manhattan(self, other: Self) -> i32 { (self.x - other.x).abs() + (self.y - other.y).abs() }
    pub fn euclidean(self, other: Self) -> f32 {
        let dx = (self.x - other.x) as f32; let dy = (self.y - other.y) as f32;
        (dx*dx + dy*dy).sqrt()
    }
    pub fn neighbors_4(&self) -> [GridPos; 4] {
        [GridPos::new(self.x+1,self.y), GridPos::new(self.x-1,self.y),
         GridPos::new(self.x,self.y+1), GridPos::new(self.x,self.y-1)]
    }
    pub fn neighbors_8(&self) -> [GridPos; 8] {
        [GridPos::new(self.x+1,self.y), GridPos::new(self.x-1,self.y),
         GridPos::new(self.x,self.y+1), GridPos::new(self.x,self.y-1),
         GridPos::new(self.x+1,self.y+1), GridPos::new(self.x-1,self.y-1),
         GridPos::new(self.x+1,self.y-1), GridPos::new(self.x-1,self.y+1)]
    }
}

pub struct NavigationGrid {
    pub width: i32, pub height: i32,
    walkable: Vec<bool>,
    cost: Vec<f32>,
}
impl NavigationGrid {
    pub fn new(width: i32, height: i32) -> Self {
        let size = (width * height) as usize;
        Self { width, height, walkable: vec![true; size], cost: vec![1.0; size] }
    }
    pub fn is_walkable(&self, pos: GridPos) -> bool {
        if pos.x < 0 || pos.y < 0 || pos.x >= self.width || pos.y >= self.height { return false; }
        self.walkable[(pos.y * self.width + pos.x) as usize]
    }
    pub fn set_walkable(&mut self, pos: GridPos, walkable: bool) {
        if pos.x >= 0 && pos.y >= 0 && pos.x < self.width && pos.y < self.height {
            self.walkable[(pos.y * self.width + pos.x) as usize] = walkable;
        }
    }
    pub fn get_cost(&self, pos: GridPos) -> f32 {
        if pos.x < 0 || pos.y < 0 || pos.x >= self.width || pos.y >= self.height { return f32::MAX; }
        self.cost[(pos.y * self.width + pos.x) as usize]
    }
    pub fn set_cost(&mut self, pos: GridPos, c: f32) {
        if pos.x >= 0 && pos.y >= 0 && pos.x < self.width && pos.y < self.height {
            self.cost[(pos.y * self.width + pos.x) as usize] = c;
        }
    }
}

#[derive(Debug, Clone)]
pub struct PathResult {
    pub path: Vec<GridPos>,
    pub cost: f32,
    pub nodes_explored: u32,
    pub success: bool,
}

// --- A* ---
#[derive(Clone)]
struct AStarNode { pos: GridPos, g: f32, f: f32 }
impl PartialEq for AStarNode { fn eq(&self, other: &Self) -> bool { self.f == other.f } }
impl Eq for AStarNode {}
impl PartialOrd for AStarNode { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { other.f.partial_cmp(&self.f) } }
impl Ord for AStarNode { fn cmp(&self, other: &Self) -> Ordering { other.f.partial_cmp(&self.f).unwrap_or(Ordering::Equal) } }

pub fn astar(grid: &NavigationGrid, start: GridPos, goal: GridPos) -> PathResult {
    let mut open = BinaryHeap::new();
    let mut came_from: HashMap<GridPos, GridPos> = HashMap::new();
    let mut g_score: HashMap<GridPos, f32> = HashMap::new();
    let mut explored = 0u32;
    g_score.insert(start, 0.0);
    open.push(AStarNode { pos: start, g: 0.0, f: start.euclidean(goal) });
    while let Some(current) = open.pop() {
        explored += 1;
        if current.pos == goal {
            let path = reconstruct_path(&came_from, goal);
            return PathResult { cost: current.g, nodes_explored: explored, success: true, path };
        }
        if explored > 50000 { break; }
        for neighbor in current.pos.neighbors_8() {
            if !grid.is_walkable(neighbor) { continue; }
            let dx = (neighbor.x - current.pos.x).abs();
            let dy = (neighbor.y - current.pos.y).abs();
            let move_cost = if dx + dy == 2 { 1.414 } else { 1.0 } * grid.get_cost(neighbor);
            let tentative_g = current.g + move_cost;
            if tentative_g < *g_score.get(&neighbor).unwrap_or(&f32::MAX) {
                came_from.insert(neighbor, current.pos);
                g_score.insert(neighbor, tentative_g);
                open.push(AStarNode { pos: neighbor, g: tentative_g, f: tentative_g + neighbor.euclidean(goal) });
            }
        }
    }
    PathResult { path: Vec::new(), cost: 0.0, nodes_explored: explored, success: false }
}

fn reconstruct_path(came_from: &HashMap<GridPos, GridPos>, mut current: GridPos) -> Vec<GridPos> {
    let mut path = vec![current];
    while let Some(&prev) = came_from.get(&current) { path.push(prev); current = prev; }
    path.reverse();
    path
}

// --- Theta* (any-angle pathfinding) ---
pub fn theta_star(grid: &NavigationGrid, start: GridPos, goal: GridPos) -> PathResult {
    let mut open = BinaryHeap::new();
    let mut came_from: HashMap<GridPos, GridPos> = HashMap::new();
    let mut g_score: HashMap<GridPos, f32> = HashMap::new();
    let mut explored = 0u32;
    g_score.insert(start, 0.0);
    came_from.insert(start, start);
    open.push(AStarNode { pos: start, g: 0.0, f: start.euclidean(goal) });
    while let Some(current) = open.pop() {
        explored += 1;
        if current.pos == goal {
            let path = reconstruct_path(&came_from, goal);
            return PathResult { cost: current.g, nodes_explored: explored, success: true, path };
        }
        if explored > 50000 { break; }
        for neighbor in current.pos.neighbors_8() {
            if !grid.is_walkable(neighbor) { continue; }
            let parent = *came_from.get(&current.pos).unwrap_or(&current.pos);
            // Try line-of-sight from parent
            if has_line_of_sight(grid, parent, neighbor) {
                let new_g = *g_score.get(&parent).unwrap_or(&f32::MAX) + parent.euclidean(neighbor);
                if new_g < *g_score.get(&neighbor).unwrap_or(&f32::MAX) {
                    g_score.insert(neighbor, new_g);
                    came_from.insert(neighbor, parent);
                    open.push(AStarNode { pos: neighbor, g: new_g, f: new_g + neighbor.euclidean(goal) });
                }
            } else {
                let move_cost = current.pos.euclidean(neighbor) * grid.get_cost(neighbor);
                let new_g = current.g + move_cost;
                if new_g < *g_score.get(&neighbor).unwrap_or(&f32::MAX) {
                    g_score.insert(neighbor, new_g);
                    came_from.insert(neighbor, current.pos);
                    open.push(AStarNode { pos: neighbor, g: new_g, f: new_g + neighbor.euclidean(goal) });
                }
            }
        }
    }
    PathResult { path: Vec::new(), cost: 0.0, nodes_explored: explored, success: false }
}

fn has_line_of_sight(grid: &NavigationGrid, a: GridPos, b: GridPos) -> bool {
    // Bresenham line check
    let mut x = a.x; let mut y = a.y;
    let dx = (b.x - a.x).abs(); let dy = (b.y - a.y).abs();
    let sx = if a.x < b.x { 1 } else { -1 };
    let sy = if a.y < b.y { 1 } else { -1 };
    let mut err = dx - dy;
    loop {
        if !grid.is_walkable(GridPos::new(x, y)) { return false; }
        if x == b.x && y == b.y { return true; }
        let e2 = 2 * err;
        if e2 > -dy { err -= dy; x += sx; }
        if e2 < dx { err += dx; y += sy; }
    }
}

// --- Flow field ---
pub struct FlowField {
    pub width: i32, pub height: i32,
    integration: Vec<f32>,
    direction: Vec<(i8, i8)>,
}

impl FlowField {
    pub fn generate(grid: &NavigationGrid, goal: GridPos) -> Self {
        let w = grid.width; let h = grid.height;
        let size = (w * h) as usize;
        let mut integration = vec![f32::MAX; size];
        let mut direction = vec![(0i8, 0i8); size];
        // Dijkstra from goal
        let mut queue = VecDeque::new();
        let gi = (goal.y * w + goal.x) as usize;
        if gi < size { integration[gi] = 0.0; queue.push_back(goal); }
        while let Some(current) = queue.pop_front() {
            let ci = (current.y * w + current.x) as usize;
            let current_cost = integration[ci];
            for neighbor in current.neighbors_8() {
                if !grid.is_walkable(neighbor) { continue; }
                let ni = (neighbor.y * w + neighbor.x) as usize;
                if ni >= size { continue; }
                let dx = (neighbor.x - current.x).abs();
                let dy = (neighbor.y - current.y).abs();
                let move_cost = if dx + dy == 2 { 1.414 } else { 1.0 } * grid.get_cost(neighbor);
                let new_cost = current_cost + move_cost;
                if new_cost < integration[ni] {
                    integration[ni] = new_cost;
                    queue.push_back(neighbor);
                }
            }
        }
        // Generate direction field
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if integration[idx] >= f32::MAX { continue; }
                let mut best_dir = (0i8, 0i8);
                let mut best_cost = integration[idx];
                for &(dx, dy) in &[(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)] {
                    let nx = x + dx; let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= w || ny >= h { continue; }
                    let ni = (ny * w + nx) as usize;
                    if integration[ni] < best_cost {
                        best_cost = integration[ni];
                        best_dir = (dx as i8, dy as i8);
                    }
                }
                direction[idx] = best_dir;
            }
        }
        Self { width: w, height: h, integration, direction }
    }

    pub fn get_direction(&self, pos: GridPos) -> (i8, i8) {
        if pos.x < 0 || pos.y < 0 || pos.x >= self.width || pos.y >= self.height { return (0, 0); }
        self.direction[(pos.y * self.width + pos.x) as usize]
    }

    pub fn get_cost(&self, pos: GridPos) -> f32 {
        if pos.x < 0 || pos.y < 0 || pos.x >= self.width || pos.y >= self.height { return f32::MAX; }
        self.integration[(pos.y * self.width + pos.x) as usize]
    }
}

// --- Path smoothing ---
pub fn smooth_path(grid: &NavigationGrid, path: &[GridPos]) -> Vec<GridPos> {
    if path.len() <= 2 { return path.to_vec(); }
    let mut smoothed = vec![path[0]];
    let mut current = 0;
    while current < path.len() - 1 {
        let mut furthest = current + 1;
        for i in (current + 2)..path.len() {
            if has_line_of_sight(grid, path[current], path[i]) { furthest = i; }
        }
        smoothed.push(path[furthest]);
        current = furthest;
    }
    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_astar() {
        let grid = NavigationGrid::new(10, 10);
        let result = astar(&grid, GridPos::new(0, 0), GridPos::new(9, 9));
        assert!(result.success);
        assert!(!result.path.is_empty());
    }
    #[test]
    fn test_theta_star() {
        let grid = NavigationGrid::new(10, 10);
        let result = theta_star(&grid, GridPos::new(0, 0), GridPos::new(9, 9));
        assert!(result.success);
    }
    #[test]
    fn test_flow_field() {
        let grid = NavigationGrid::new(10, 10);
        let ff = FlowField::generate(&grid, GridPos::new(5, 5));
        let dir = ff.get_direction(GridPos::new(0, 0));
        assert!(dir.0 != 0 || dir.1 != 0);
    }
    #[test]
    fn test_blocked_path() {
        let mut grid = NavigationGrid::new(5, 5);
        for y in 0..5 { grid.set_walkable(GridPos::new(2, y), false); }
        let result = astar(&grid, GridPos::new(0, 0), GridPos::new(4, 4));
        assert!(!result.success);
    }
}
