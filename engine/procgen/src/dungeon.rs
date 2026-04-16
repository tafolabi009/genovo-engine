//! # Dungeon Generation
//!
//! Multiple algorithms for procedural dungeon generation:
//!
//! - **BSP (Binary Space Partition)**: Recursively subdivide a rectangle into
//!   rooms, then connect them with corridors.
//! - **Cellular automata caves**: Random fill followed by iterative smoothing
//!   to create organic cave systems.
//! - **Drunkard's walk**: Random walk carving to create winding cave passages.
//! - **Room placement + MST corridors**: Place random rooms, connect them via
//!   minimum spanning tree for guaranteed connectivity.
//!
//! All algorithms output a `DungeonMap` containing tile data, room info, and
//! corridor data. Post-processing adds doors, stairs, and spawn points.

use genovo_core::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

// ===========================================================================
// Tile types
// ===========================================================================

/// Tile types that can appear in a generated dungeon map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DungeonTile {
    /// Impassable wall.
    Wall,
    /// Walkable floor.
    Floor,
    /// Door between a room and corridor.
    Door,
    /// Upward staircase (entrance).
    StairsUp,
    /// Downward staircase (exit).
    StairsDown,
    /// Treasure chest location.
    Chest,
    /// Enemy spawn point.
    Enemy,
    /// A special marker for the player start.
    PlayerStart,
    /// Water/lava hazard tile.
    Hazard,
    /// A secret passage (appears as wall but is passable).
    SecretPassage,
}

impl DungeonTile {
    /// Whether this tile is walkable by default.
    pub fn is_walkable(self) -> bool {
        matches!(
            self,
            DungeonTile::Floor
                | DungeonTile::Door
                | DungeonTile::StairsUp
                | DungeonTile::StairsDown
                | DungeonTile::Chest
                | DungeonTile::Enemy
                | DungeonTile::PlayerStart
                | DungeonTile::SecretPassage
        )
    }

    /// Whether this tile blocks movement.
    pub fn is_solid(self) -> bool {
        matches!(self, DungeonTile::Wall)
    }
}

// ===========================================================================
// Room
// ===========================================================================

/// A rectangular room in the dungeon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    /// Left edge (inclusive).
    pub x: usize,
    /// Top edge (inclusive).
    pub y: usize,
    /// Room width.
    pub width: usize,
    /// Room height.
    pub height: usize,
    /// Unique room identifier.
    pub id: usize,
    /// Center point of the room.
    pub center: (usize, usize),
    /// Room tags (e.g., "boss", "treasure", "start").
    pub tags: Vec<String>,
}

impl Room {
    /// Create a new room with the given bounds.
    pub fn new(id: usize, x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
            id,
            center: (x + width / 2, y + height / 2),
            tags: Vec::new(),
        }
    }

    /// Right edge (exclusive).
    pub fn right(&self) -> usize {
        self.x + self.width
    }

    /// Bottom edge (exclusive).
    pub fn bottom(&self) -> usize {
        self.y + self.height
    }

    /// Area of the room in tiles.
    pub fn area(&self) -> usize {
        self.width * self.height
    }

    /// Check if this room overlaps another room (with optional padding).
    pub fn overlaps(&self, other: &Room, padding: usize) -> bool {
        let ax1 = self.x.saturating_sub(padding);
        let ay1 = self.y.saturating_sub(padding);
        let ax2 = self.right() + padding;
        let ay2 = self.bottom() + padding;

        let bx1 = other.x.saturating_sub(padding);
        let by1 = other.y.saturating_sub(padding);
        let bx2 = other.right() + padding;
        let by2 = other.bottom() + padding;

        ax1 < bx2 && ax2 > bx1 && ay1 < by2 && ay2 > by1
    }

    /// Check if a point is inside the room.
    pub fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.x && x < self.right() && y >= self.y && y < self.bottom()
    }

    /// Add a tag to this room.
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Distance to another room's center.
    pub fn distance_to(&self, other: &Room) -> f32 {
        let dx = self.center.0 as f32 - other.center.0 as f32;
        let dy = self.center.1 as f32 - other.center.1 as f32;
        (dx * dx + dy * dy).sqrt()
    }
}

// ===========================================================================
// Corridor
// ===========================================================================

/// A corridor connecting two rooms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Corridor {
    /// Ordered list of tile positions forming the corridor.
    pub tiles: Vec<(usize, usize)>,
    /// Index of the source room.
    pub from_room: usize,
    /// Index of the destination room.
    pub to_room: usize,
}

// ===========================================================================
// DungeonMap
// ===========================================================================

/// The output of a dungeon generation algorithm.
///
/// Contains the tile grid, room info, corridor info, and special locations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DungeonMap {
    /// Map width.
    pub width: usize,
    /// Map height.
    pub height: usize,
    /// Flat tile array, indexed as `y * width + x`.
    pub tiles: Vec<DungeonTile>,
    /// List of rooms in the dungeon.
    pub rooms: Vec<Room>,
    /// List of corridors connecting rooms.
    pub corridors: Vec<Corridor>,
    /// Player start position.
    pub player_start: Option<(usize, usize)>,
    /// Exit position (stairs down).
    pub exit: Option<(usize, usize)>,
    /// Dijkstra distance map from the entrance (for difficulty scaling).
    pub difficulty_map: Vec<i32>,
}

impl DungeonMap {
    /// Create a new dungeon map filled with walls.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            tiles: vec![DungeonTile::Wall; width * height],
            rooms: Vec::new(),
            corridors: Vec::new(),
            player_start: None,
            exit: None,
            difficulty_map: vec![-1; width * height],
        }
    }

    /// Get the tile at (x, y).
    pub fn get(&self, x: usize, y: usize) -> DungeonTile {
        if x < self.width && y < self.height {
            self.tiles[y * self.width + x]
        } else {
            DungeonTile::Wall
        }
    }

    /// Set the tile at (x, y).
    pub fn set(&mut self, x: usize, y: usize, tile: DungeonTile) {
        if x < self.width && y < self.height {
            self.tiles[y * self.width + x] = tile;
        }
    }

    /// Safe get with bounds check.
    pub fn get_safe(&self, x: i32, y: i32) -> DungeonTile {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.tiles[y as usize * self.width + x as usize]
        } else {
            DungeonTile::Wall
        }
    }

    /// Count the number of wall neighbors around (x, y) in a Moore neighborhood.
    pub fn count_wall_neighbors(&self, x: usize, y: usize) -> usize {
        let mut count = 0;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if self.get_safe(nx, ny) == DungeonTile::Wall {
                    count += 1;
                }
            }
        }
        count
    }

    /// Count cardinal (Von Neumann) wall neighbors.
    pub fn count_cardinal_wall_neighbors(&self, x: usize, y: usize) -> usize {
        let mut count = 0;
        for &(dx, dy) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if self.get_safe(nx, ny) == DungeonTile::Wall {
                count += 1;
            }
        }
        count
    }

    /// Carve out a rectangular area as floor.
    pub fn carve_room(&mut self, room: &Room) {
        for y in room.y..room.bottom().min(self.height) {
            for x in room.x..room.right().min(self.width) {
                self.set(x, y, DungeonTile::Floor);
            }
        }
    }

    /// Carve a horizontal tunnel from x1 to x2 at y.
    pub fn carve_h_tunnel(&mut self, x1: usize, x2: usize, y: usize) -> Vec<(usize, usize)> {
        let mut tiles = Vec::new();
        let start = x1.min(x2);
        let end = x1.max(x2);
        for x in start..=end {
            if x < self.width && y < self.height {
                self.set(x, y, DungeonTile::Floor);
                tiles.push((x, y));
            }
        }
        tiles
    }

    /// Carve a vertical tunnel from y1 to y2 at x.
    pub fn carve_v_tunnel(&mut self, y1: usize, y2: usize, x: usize) -> Vec<(usize, usize)> {
        let mut tiles = Vec::new();
        let start = y1.min(y2);
        let end = y1.max(y2);
        for y in start..=end {
            if x < self.width && y < self.height {
                self.set(x, y, DungeonTile::Floor);
                tiles.push((x, y));
            }
        }
        tiles
    }

    /// Carve an L-shaped corridor between two points.
    pub fn carve_l_corridor(
        &mut self,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
        horizontal_first: bool,
    ) -> Vec<(usize, usize)> {
        let mut tiles = Vec::new();
        if horizontal_first {
            tiles.extend(self.carve_h_tunnel(x1, x2, y1));
            tiles.extend(self.carve_v_tunnel(y1, y2, x2));
        } else {
            tiles.extend(self.carve_v_tunnel(y1, y2, x1));
            tiles.extend(self.carve_h_tunnel(x1, x2, y2));
        }
        tiles
    }

    /// Find the room that contains a given point.
    pub fn room_at(&self, x: usize, y: usize) -> Option<usize> {
        for (i, room) in self.rooms.iter().enumerate() {
            if room.contains(x, y) {
                return Some(i);
            }
        }
        None
    }

    /// Post-processing: place doors at corridor-room boundaries.
    pub fn place_doors(&mut self) {
        let width = self.width;
        let height = self.height;

        // Collect corridor tile positions to avoid borrowing self.corridors
        // while mutating self.tiles.
        let corridor_tiles: Vec<(usize, usize)> = self
            .corridors
            .iter()
            .flat_map(|c| c.tiles.iter().copied())
            .collect();

        for (cx, cy) in corridor_tiles {
            // Check if this corridor tile is adjacent to a room wall.
            for &(dx, dy) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                    continue;
                }
                // If neighbor is a wall and the cell beyond it is a room floor,
                // this is a boundary.
                let ux = nx as usize;
                let uy = ny as usize;
                if self.get(ux, uy) == DungeonTile::Wall {
                    let bx = nx + dx;
                    let by = ny + dy;
                    if bx >= 0
                        && by >= 0
                        && (bx as usize) < width
                        && (by as usize) < height
                        && self.get(bx as usize, by as usize) == DungeonTile::Floor
                        && self.room_at(bx as usize, by as usize).is_some()
                    {
                        // Place a door at the corridor tile if it has exactly
                        // 2 wall neighbors in cardinal directions.
                        let wall_count = self.count_cardinal_wall_neighbors(cx, cy);
                        if wall_count == 2 {
                            self.set(cx, cy, DungeonTile::Door);
                        }
                    }
                }
            }
        }
    }

    /// Post-processing: place stairs in the first and last rooms.
    pub fn place_stairs(&mut self) {
        if self.rooms.len() < 2 {
            return;
        }

        // Extract room centers before mutating.
        let start_center = self.rooms[0].center;
        let end_center = self.rooms[self.rooms.len() - 1].center;

        // Place stairs up (entrance) in the first room.
        self.set(start_center.0, start_center.1, DungeonTile::StairsUp);
        self.player_start = Some(start_center);

        // Place stairs down (exit) in the last room.
        self.set(end_center.0, end_center.1, DungeonTile::StairsDown);
        self.exit = Some(end_center);
    }

    /// Build a Dijkstra distance map from the player start position.
    ///
    /// This is useful for difficulty scaling: place harder enemies further
    /// from the entrance.
    pub fn build_difficulty_map(&mut self) {
        let start = match self.player_start {
            Some(s) => s,
            None => return,
        };

        self.difficulty_map = vec![-1i32; self.width * self.height];
        self.difficulty_map[start.1 * self.width + start.0] = 0;

        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some((x, y)) = queue.pop_front() {
            let current_dist = self.difficulty_map[y * self.width + x];

            for &(dx, dy) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx as usize >= self.width || ny as usize >= self.height {
                    continue;
                }
                let ux = nx as usize;
                let uy = ny as usize;
                let idx = uy * self.width + ux;
                if self.tiles[idx].is_walkable() && self.difficulty_map[idx] == -1 {
                    self.difficulty_map[idx] = current_dist + 1;
                    queue.push_back((ux, uy));
                }
            }
        }
    }

    /// Place enemies and chests based on the difficulty map.
    ///
    /// Enemies are placed in rooms proportional to their distance from
    /// the entrance. Chests are placed in dead-end rooms.
    pub fn populate(&mut self, rng: &mut Rng, enemy_density: f32, chest_density: f32) {
        let rooms = self.rooms.clone();

        for (i, room) in rooms.iter().enumerate() {
            // Skip the start and end rooms.
            if i == 0 || i == rooms.len() - 1 {
                continue;
            }

            // Get the average difficulty in this room.
            let mut total_diff = 0i32;
            let mut count = 0;
            for y in room.y..room.bottom().min(self.height) {
                for x in room.x..room.right().min(self.width) {
                    let d = self.difficulty_map[y * self.width + x];
                    if d > 0 {
                        total_diff += d;
                        count += 1;
                    }
                }
            }

            if count == 0 {
                continue;
            }
            let avg_diff = total_diff as f32 / count as f32;

            // Scale enemy count by difficulty and room size.
            let difficulty_factor = (avg_diff / 20.0).min(1.0) + 0.2;
            let enemy_count =
                (room.area() as f32 * enemy_density * difficulty_factor).round() as usize;
            let chest_count = (room.area() as f32 * chest_density * 0.5).round() as usize;

            // Place enemies at random floor positions.
            let mut floor_positions: Vec<(usize, usize)> = Vec::new();
            for y in room.y..room.bottom().min(self.height) {
                for x in room.x..room.right().min(self.width) {
                    if self.get(x, y) == DungeonTile::Floor {
                        floor_positions.push((x, y));
                    }
                }
            }

            rng.shuffle(&mut floor_positions);

            for (j, &(x, y)) in floor_positions.iter().enumerate() {
                if j < enemy_count {
                    self.set(x, y, DungeonTile::Enemy);
                } else if j < enemy_count + chest_count {
                    self.set(x, y, DungeonTile::Chest);
                } else {
                    break;
                }
            }
        }
    }

    /// Render the dungeon as a string using ASCII art.
    pub fn to_ascii(&self) -> String {
        let mut result = String::with_capacity(self.width * self.height * 2);
        for y in 0..self.height {
            for x in 0..self.width {
                let ch = match self.get(x, y) {
                    DungeonTile::Wall => '#',
                    DungeonTile::Floor => '.',
                    DungeonTile::Door => '+',
                    DungeonTile::StairsUp => '<',
                    DungeonTile::StairsDown => '>',
                    DungeonTile::Chest => '$',
                    DungeonTile::Enemy => 'E',
                    DungeonTile::PlayerStart => '@',
                    DungeonTile::Hazard => '~',
                    DungeonTile::SecretPassage => 'S',
                };
                result.push(ch);
            }
            result.push('\n');
        }
        result
    }

    /// Count the total number of floor tiles (walkable area).
    pub fn floor_count(&self) -> usize {
        self.tiles.iter().filter(|t| t.is_walkable()).count()
    }

    /// Count the total number of wall tiles.
    pub fn wall_count(&self) -> usize {
        self.tiles.iter().filter(|t| t.is_solid()).count()
    }
}

// ===========================================================================
// BSP Dungeon Generation
// ===========================================================================

/// Configuration for BSP dungeon generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BSPConfig {
    /// Map width.
    pub width: usize,
    /// Map height.
    pub height: usize,
    /// Minimum room size (width or height).
    pub min_room_size: usize,
    /// Maximum depth of BSP tree subdivision.
    pub max_depth: u32,
    /// Minimum ratio of room to partition (0.0-1.0). Higher values make rooms
    /// fill more of their partition.
    pub room_fill_ratio: f32,
    /// Random seed.
    pub seed: u64,
    /// Wall padding around rooms (minimum gap between rooms).
    pub wall_padding: usize,
}

impl Default for BSPConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 50,
            min_room_size: 5,
            max_depth: 5,
            room_fill_ratio: 0.6,
            seed: 42,
            wall_padding: 1,
        }
    }
}

/// A node in the BSP tree.
#[derive(Debug, Clone)]
struct BSPNode {
    /// Left edge of the partition.
    x: usize,
    /// Top edge of the partition.
    y: usize,
    /// Partition width.
    width: usize,
    /// Partition height.
    height: usize,
    /// Left/top child node.
    left: Option<Box<BSPNode>>,
    /// Right/bottom child node.
    right: Option<Box<BSPNode>>,
    /// Room carved in this node (only for leaves).
    room: Option<Room>,
    /// Depth in the BSP tree.
    depth: u32,
}

impl BSPNode {
    fn new(x: usize, y: usize, width: usize, height: usize, depth: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            left: None,
            right: None,
            room: None,
            depth,
        }
    }

    /// Recursively split this node into two children.
    fn split(&mut self, rng: &mut Rng, min_size: usize, max_depth: u32, room_id: &mut usize, config: &BSPConfig) {
        if self.depth >= max_depth {
            self.create_room(rng, min_size, room_id, config);
            return;
        }

        // Decide split direction: prefer splitting the longer axis.
        let split_horizontal = if self.width > self.height * 2 {
            false // Split vertically (left/right).
        } else if self.height > self.width * 2 {
            true // Split horizontally (top/bottom).
        } else {
            rng.bool(0.5)
        };

        if split_horizontal {
            // Horizontal split: top and bottom halves.
            let min_y = min_size + config.wall_padding;
            let max_y = self.height.saturating_sub(min_size + config.wall_padding);
            if min_y >= max_y || max_y <= 1 {
                self.create_room(rng, min_size, room_id, config);
                return;
            }
            let split_y = rng.range_i32(min_y as i32, max_y as i32) as usize;

            let mut top = BSPNode::new(self.x, self.y, self.width, split_y, self.depth + 1);
            let mut bottom = BSPNode::new(
                self.x,
                self.y + split_y,
                self.width,
                self.height - split_y,
                self.depth + 1,
            );

            top.split(rng, min_size, max_depth, room_id, config);
            bottom.split(rng, min_size, max_depth, room_id, config);

            self.left = Some(Box::new(top));
            self.right = Some(Box::new(bottom));
        } else {
            // Vertical split: left and right halves.
            let min_x = min_size + config.wall_padding;
            let max_x = self.width.saturating_sub(min_size + config.wall_padding);
            if min_x >= max_x || max_x <= 1 {
                self.create_room(rng, min_size, room_id, config);
                return;
            }
            let split_x = rng.range_i32(min_x as i32, max_x as i32) as usize;

            let mut left = BSPNode::new(self.x, self.y, split_x, self.height, self.depth + 1);
            let mut right = BSPNode::new(
                self.x + split_x,
                self.y,
                self.width - split_x,
                self.height,
                self.depth + 1,
            );

            left.split(rng, min_size, max_depth, room_id, config);
            right.split(rng, min_size, max_depth, room_id, config);

            self.left = Some(Box::new(left));
            self.right = Some(Box::new(right));
        }
    }

    /// Create a room within this leaf node.
    fn create_room(&mut self, rng: &mut Rng, min_size: usize, room_id: &mut usize, config: &BSPConfig) {
        let padding = config.wall_padding;
        let max_w = self.width.saturating_sub(padding * 2);
        let max_h = self.height.saturating_sub(padding * 2);

        if max_w < min_size || max_h < min_size {
            return;
        }

        let min_w = (max_w as f32 * config.room_fill_ratio).max(min_size as f32) as usize;
        let min_h = (max_h as f32 * config.room_fill_ratio).max(min_size as f32) as usize;

        let w = rng.range_i32(min_w.min(max_w) as i32, (max_w + 1) as i32) as usize;
        let h = rng.range_i32(min_h.min(max_h) as i32, (max_h + 1) as i32) as usize;

        let rx = self.x + padding + rng.range_i32(0, (max_w - w + 1) as i32) as usize;
        let ry = self.y + padding + rng.range_i32(0, (max_h - h + 1) as i32) as usize;

        self.room = Some(Room::new(*room_id, rx, ry, w, h));
        *room_id += 1;
    }

    /// Get the room for this node (or a room from a descendant).
    fn get_room(&self) -> Option<&Room> {
        if self.room.is_some() {
            return self.room.as_ref();
        }
        // Try left child first, then right.
        if let Some(ref left) = self.left {
            if let Some(room) = left.get_room() {
                return Some(room);
            }
        }
        if let Some(ref right) = self.right {
            if let Some(room) = right.get_room() {
                return Some(room);
            }
        }
        None
    }

    /// Collect all rooms from leaf nodes.
    fn collect_rooms(&self, rooms: &mut Vec<Room>) {
        if let Some(ref room) = self.room {
            rooms.push(room.clone());
        }
        if let Some(ref left) = self.left {
            left.collect_rooms(rooms);
        }
        if let Some(ref right) = self.right {
            right.collect_rooms(rooms);
        }
    }

    /// Connect sibling rooms with corridors.
    fn connect_rooms(&self, map: &mut DungeonMap, rng: &mut Rng) {
        if let (Some(left), Some(right)) = (&self.left, &self.right) {
            // Connect a room from the left subtree to a room from the right subtree.
            if let (Some(lr), Some(rr)) = (left.get_room(), right.get_room()) {
                let (x1, y1) = lr.center;
                let (x2, y2) = rr.center;

                // Choose L-shaped or straight corridor.
                let horizontal_first = rng.bool(0.5);
                let tiles = map.carve_l_corridor(x1, y1, x2, y2, horizontal_first);

                map.corridors.push(Corridor {
                    tiles,
                    from_room: lr.id,
                    to_room: rr.id,
                });
            }

            // Recurse into children to connect their sub-partitions.
            left.connect_rooms(map, rng);
            right.connect_rooms(map, rng);
        }
    }
}

/// Generate a dungeon using Binary Space Partition.
///
/// The algorithm recursively subdivides the map into partitions, places a room
/// in each leaf partition, and connects sibling rooms with L-shaped corridors.
pub fn generate_bsp(config: &BSPConfig) -> DungeonMap {
    let mut rng = Rng::new(config.seed);
    let mut map = DungeonMap::new(config.width, config.height);
    let mut room_id = 0;

    let mut root = BSPNode::new(0, 0, config.width, config.height, 0);
    root.split(&mut rng, config.min_room_size, config.max_depth, &mut room_id, config);

    // Collect all rooms from the BSP tree.
    let mut rooms = Vec::new();
    root.collect_rooms(&mut rooms);

    // Carve rooms into the map.
    for room in &rooms {
        map.carve_room(room);
    }

    map.rooms = rooms;

    // Connect rooms via the BSP tree hierarchy.
    root.connect_rooms(&mut map, &mut rng);

    // Post-processing.
    map.place_stairs();
    map.place_doors();
    map.build_difficulty_map();

    map
}

// ===========================================================================
// Cellular Automata Caves
// ===========================================================================

/// Configuration for cellular automata cave generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaveConfig {
    /// Map width.
    pub width: usize,
    /// Map height.
    pub height: usize,
    /// Initial fill percentage (0-100). Higher values produce more walls.
    pub fill_percent: u32,
    /// Number of smoothing iterations.
    pub iterations: u32,
    /// Threshold for the smoothing rule: if a cell has >= this many wall
    /// neighbors, it becomes a wall; otherwise it becomes floor.
    pub wall_threshold: usize,
    /// Minimum cave region size. Regions smaller than this are filled in.
    pub min_region_size: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for CaveConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 50,
            fill_percent: 45,
            iterations: 5,
            wall_threshold: 5,
            min_region_size: 20,
            seed: 42,
        }
    }
}

/// Generate a cave using cellular automata.
///
/// 1. Initialize grid with random fill based on `fill_percent`.
/// 2. For each iteration, apply the smoothing rule: if a cell has >= `wall_threshold`
///    wall neighbors (Moore neighborhood), it becomes a wall; otherwise floor.
/// 3. Find the largest connected floor region via flood fill.
/// 4. Remove smaller regions by filling them with walls.
pub fn generate_cave(config: &CaveConfig) -> DungeonMap {
    let mut rng = Rng::new(config.seed);
    let mut map = DungeonMap::new(config.width, config.height);

    // Step 1: Random fill.
    for y in 0..config.height {
        for x in 0..config.width {
            // Always make borders walls.
            if x == 0 || x == config.width - 1 || y == 0 || y == config.height - 1 {
                map.set(x, y, DungeonTile::Wall);
            } else if rng.range_i32(0, 100) < config.fill_percent as i32 {
                map.set(x, y, DungeonTile::Wall);
            } else {
                map.set(x, y, DungeonTile::Floor);
            }
        }
    }

    // Step 2: Smoothing iterations.
    for _ in 0..config.iterations {
        let old_tiles = map.tiles.clone();

        for y in 1..config.height - 1 {
            for x in 1..config.width - 1 {
                let mut wall_count = 0;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = (x as i32 + dx) as usize;
                        let ny = (y as i32 + dy) as usize;
                        if old_tiles[ny * config.width + nx] == DungeonTile::Wall {
                            wall_count += 1;
                        }
                    }
                }

                if wall_count >= config.wall_threshold {
                    map.set(x, y, DungeonTile::Wall);
                } else {
                    map.set(x, y, DungeonTile::Floor);
                }
            }
        }
    }

    // Step 3: Find connected regions via flood fill.
    let regions = find_floor_regions(&map);

    if regions.is_empty() {
        return map;
    }

    // Step 4: Keep only the largest region, fill others.
    let largest_idx = regions
        .iter()
        .enumerate()
        .max_by_key(|(_, r)| r.len())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let largest_region: HashSet<(usize, usize)> =
        regions[largest_idx].iter().cloned().collect();

    // Fill all floor tiles not in the largest region.
    for y in 0..config.height {
        for x in 0..config.width {
            if map.get(x, y) == DungeonTile::Floor && !largest_region.contains(&(x, y)) {
                map.set(x, y, DungeonTile::Wall);
            }
        }
    }

    // Also remove small regions.
    for (i, region) in regions.iter().enumerate() {
        if i == largest_idx {
            continue;
        }
        if region.len() < config.min_region_size {
            for &(x, y) in region {
                map.set(x, y, DungeonTile::Wall);
            }
        }
    }

    // Create "rooms" from contiguous floor areas for the room list.
    // For caves, we create one big room representing the cave.
    let mut min_x = config.width;
    let mut min_y = config.height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;

    for &(x, y) in &regions[largest_idx] {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    let cave_room = Room::new(0, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
    map.rooms.push(cave_room);

    // Place entrance and exit at extremes of the cave.
    let floor_tiles: Vec<(usize, usize)> = regions[largest_idx].clone();
    if floor_tiles.len() >= 2 {
        // Place stairs at the two furthest floor tiles from each other.
        let start = floor_tiles[0];
        map.set(start.0, start.1, DungeonTile::StairsUp);
        map.player_start = Some(start);

        // Find the tile furthest from the start using BFS.
        let mut dist_map = vec![-1i32; config.width * config.height];
        dist_map[start.1 * config.width + start.0] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        let mut furthest = start;
        let mut max_dist = 0i32;

        while let Some((x, y)) = queue.pop_front() {
            let cd = dist_map[y * config.width + x];
            for &(dx, dy) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 {
                    let ux = nx as usize;
                    let uy = ny as usize;
                    if ux < config.width && uy < config.height {
                        let idx = uy * config.width + ux;
                        if map.tiles[idx] == DungeonTile::Floor && dist_map[idx] == -1 {
                            dist_map[idx] = cd + 1;
                            queue.push_back((ux, uy));
                            if cd + 1 > max_dist {
                                max_dist = cd + 1;
                                furthest = (ux, uy);
                            }
                        }
                    }
                }
            }
        }

        map.set(furthest.0, furthest.1, DungeonTile::StairsDown);
        map.exit = Some(furthest);
    }

    map.build_difficulty_map();
    map
}

/// Find all connected floor regions using flood fill.
fn find_floor_regions(map: &DungeonMap) -> Vec<Vec<(usize, usize)>> {
    let mut visited = vec![false; map.width * map.height];
    let mut regions = Vec::new();

    for y in 0..map.height {
        for x in 0..map.width {
            let idx = y * map.width + x;
            if map.tiles[idx] == DungeonTile::Floor && !visited[idx] {
                let region = flood_fill(map, x, y, &mut visited);
                if !region.is_empty() {
                    regions.push(region);
                }
            }
        }
    }

    regions
}

/// Flood fill from a starting position, returning all connected floor tiles.
fn flood_fill(
    map: &DungeonMap,
    start_x: usize,
    start_y: usize,
    visited: &mut [bool],
) -> Vec<(usize, usize)> {
    let mut region = Vec::new();
    let mut queue = VecDeque::new();

    let start_idx = start_y * map.width + start_x;
    if visited[start_idx] || map.tiles[start_idx] != DungeonTile::Floor {
        return region;
    }

    visited[start_idx] = true;
    queue.push_back((start_x, start_y));

    while let Some((x, y)) = queue.pop_front() {
        region.push((x, y));

        for &(dx, dy) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 {
                let ux = nx as usize;
                let uy = ny as usize;
                if ux < map.width && uy < map.height {
                    let idx = uy * map.width + ux;
                    if !visited[idx] && map.tiles[idx] == DungeonTile::Floor {
                        visited[idx] = true;
                        queue.push_back((ux, uy));
                    }
                }
            }
        }
    }

    region
}

// ===========================================================================
// Drunkard's Walk
// ===========================================================================

/// Configuration for drunkard's walk cave generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrunkardConfig {
    /// Map width.
    pub width: usize,
    /// Map height.
    pub height: usize,
    /// Target percentage of floor tiles (0.0-1.0).
    pub target_floor_percent: f32,
    /// Maximum number of steps before giving up.
    pub max_steps: usize,
    /// Whether to start from the center.
    pub start_center: bool,
    /// Whether to weight the walk toward unexplored areas.
    pub weighted_toward_open: bool,
    /// Random seed.
    pub seed: u64,
}

impl Default for DrunkardConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 50,
            target_floor_percent: 0.4,
            max_steps: 100_000,
            start_center: true,
            weighted_toward_open: false,
            seed: 42,
        }
    }
}

/// Generate a cave using the drunkard's walk algorithm.
///
/// Starting from a point (usually the center), a random walker moves in
/// cardinal directions, carving floor tiles as it goes. The walk continues
/// until the target floor percentage is reached.
pub fn generate_drunkard(config: &DrunkardConfig) -> DungeonMap {
    let mut rng = Rng::new(config.seed);
    let mut map = DungeonMap::new(config.width, config.height);
    let total_tiles = config.width * config.height;
    let target_floor = (total_tiles as f32 * config.target_floor_percent) as usize;

    // Starting position.
    let mut x = if config.start_center {
        config.width / 2
    } else {
        rng.range_i32(1, config.width as i32 - 1) as usize
    };
    let mut y = if config.start_center {
        config.height / 2
    } else {
        rng.range_i32(1, config.height as i32 - 1) as usize
    };

    let start = (x, y);
    map.set(x, y, DungeonTile::Floor);
    let mut floor_count = 1;
    let mut steps = 0;

    let directions: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

    while floor_count < target_floor && steps < config.max_steps {
        steps += 1;

        // Pick a random direction.
        let dir_idx = if config.weighted_toward_open {
            // Weight toward directions with more walls (to explore new area).
            let mut weights = [0.0f32; 4];
            for (i, &(dx, dy)) in directions.iter().enumerate() {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if map.get_safe(nx, ny) == DungeonTile::Wall {
                    weights[i] = 2.0;
                } else {
                    weights[i] = 1.0;
                }
            }
            rng.weighted_pick(&weights)
        } else {
            rng.range_i32(0, 4) as usize
        };

        let (dx, dy) = directions[dir_idx];
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;

        // Stay within bounds (leave 1-tile border).
        if nx > 0
            && ny > 0
            && (nx as usize) < config.width - 1
            && (ny as usize) < config.height - 1
        {
            x = nx as usize;
            y = ny as usize;

            if map.get(x, y) == DungeonTile::Wall {
                map.set(x, y, DungeonTile::Floor);
                floor_count += 1;
            }
        }
    }

    // Place stairs at start and find furthest point for exit.
    map.set(start.0, start.1, DungeonTile::StairsUp);
    map.player_start = Some(start);

    // Find the furthest reachable floor tile.
    let furthest = find_furthest_floor(&map, start);
    if let Some(exit) = furthest {
        map.set(exit.0, exit.1, DungeonTile::StairsDown);
        map.exit = Some(exit);
    }

    // Create a single cave room.
    let room = Room::new(0, 1, 1, config.width - 2, config.height - 2);
    map.rooms.push(room);

    map.build_difficulty_map();
    map
}

/// Find the floor tile furthest from a starting point using BFS.
fn find_furthest_floor(map: &DungeonMap, start: (usize, usize)) -> Option<(usize, usize)> {
    let mut dist = vec![-1i32; map.width * map.height];
    let mut queue = VecDeque::new();

    dist[start.1 * map.width + start.0] = 0;
    queue.push_back(start);

    let mut furthest = start;
    let mut max_dist = 0;

    while let Some((x, y)) = queue.pop_front() {
        let cd = dist[y * map.width + x];

        for &(dx, dy) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 {
                let ux = nx as usize;
                let uy = ny as usize;
                if ux < map.width && uy < map.height {
                    let idx = uy * map.width + ux;
                    if map.tiles[idx].is_walkable() && dist[idx] == -1 {
                        dist[idx] = cd + 1;
                        queue.push_back((ux, uy));
                        if cd + 1 > max_dist {
                            max_dist = cd + 1;
                            furthest = (ux, uy);
                        }
                    }
                }
            }
        }
    }

    if max_dist > 0 {
        Some(furthest)
    } else {
        None
    }
}

// ===========================================================================
// Room Placement + MST Corridors
// ===========================================================================

/// Configuration for room-placement dungeon generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomPlacementConfig {
    /// Map width.
    pub width: usize,
    /// Map height.
    pub height: usize,
    /// Number of rooms to attempt to place.
    pub room_count: usize,
    /// Minimum room width.
    pub min_room_width: usize,
    /// Maximum room width.
    pub max_room_width: usize,
    /// Minimum room height.
    pub min_room_height: usize,
    /// Maximum room height.
    pub max_room_height: usize,
    /// Padding between rooms.
    pub room_padding: usize,
    /// Probability of adding extra corridors beyond the MST (0.0-1.0).
    /// Higher values create more loops.
    pub extra_corridor_chance: f32,
    /// Random seed.
    pub seed: u64,
}

impl Default for RoomPlacementConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 50,
            room_count: 10,
            min_room_width: 5,
            max_room_width: 15,
            min_room_height: 4,
            max_room_height: 12,
            room_padding: 2,
            extra_corridor_chance: 0.15,
            seed: 42,
        }
    }
}

/// Edge in a graph (for MST construction).
#[derive(Debug, Clone, PartialEq)]
struct Edge {
    from: usize,
    to: usize,
    weight: f32,
}

impl Eq for Edge {}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap behavior in BinaryHeap (which is a max-heap).
        other
            .weight
            .partial_cmp(&self.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Generate a dungeon by placing random rooms and connecting them with MST corridors.
///
/// 1. Place N non-overlapping rooms of random size.
/// 2. Build the minimum spanning tree of room centers (Prim's algorithm).
/// 3. Dig corridors along MST edges.
/// 4. Optionally add extra random connections for loops.
pub fn generate_room_placement(config: &RoomPlacementConfig) -> DungeonMap {
    let mut rng = Rng::new(config.seed);
    let mut map = DungeonMap::new(config.width, config.height);
    let mut rooms: Vec<Room> = Vec::new();

    // Step 1: Place rooms.
    let max_attempts = config.room_count * 50;
    let mut attempts = 0;

    while rooms.len() < config.room_count && attempts < max_attempts {
        attempts += 1;

        let w = rng.range_i32(config.min_room_width as i32, (config.max_room_width + 1) as i32)
            as usize;
        let h = rng.range_i32(
            config.min_room_height as i32,
            (config.max_room_height + 1) as i32,
        ) as usize;

        let max_x = config.width.saturating_sub(w + 1);
        let max_y = config.height.saturating_sub(h + 1);
        if max_x <= 1 || max_y <= 1 {
            continue;
        }

        let x = rng.range_i32(1, max_x as i32) as usize;
        let y = rng.range_i32(1, max_y as i32) as usize;

        let new_room = Room::new(rooms.len(), x, y, w, h);

        // Check overlap with existing rooms.
        let overlaps = rooms
            .iter()
            .any(|r| new_room.overlaps(r, config.room_padding));

        if !overlaps {
            map.carve_room(&new_room);
            rooms.push(new_room);
        }
    }

    if rooms.len() < 2 {
        map.rooms = rooms;
        return map;
    }

    // Step 2: Build MST using Prim's algorithm.
    let n = rooms.len();
    let mut in_mst = vec![false; n];
    let mut mst_edges: Vec<(usize, usize)> = Vec::new();
    let mut heap: BinaryHeap<Edge> = BinaryHeap::new();

    // Start from room 0.
    in_mst[0] = true;
    for j in 1..n {
        let dist = rooms[0].distance_to(&rooms[j]);
        heap.push(Edge {
            from: 0,
            to: j,
            weight: dist,
        });
    }

    while let Some(edge) = heap.pop() {
        if in_mst[edge.to] {
            continue;
        }
        in_mst[edge.to] = true;
        mst_edges.push((edge.from, edge.to));

        // Add edges from the newly added node.
        for j in 0..n {
            if !in_mst[j] {
                let dist = rooms[edge.to].distance_to(&rooms[j]);
                heap.push(Edge {
                    from: edge.to,
                    to: j,
                    weight: dist,
                });
            }
        }
    }

    // Step 3: Dig corridors along MST edges.
    for &(from, to) in &mst_edges {
        let (x1, y1) = rooms[from].center;
        let (x2, y2) = rooms[to].center;
        let horizontal_first = rng.bool(0.5);
        let tiles = map.carve_l_corridor(x1, y1, x2, y2, horizontal_first);
        map.corridors.push(Corridor {
            tiles,
            from_room: from,
            to_room: to,
        });
    }

    // Step 4: Add extra corridors for loops.
    for i in 0..n {
        for j in (i + 1)..n {
            // Skip edges already in the MST.
            if mst_edges.contains(&(i, j)) || mst_edges.contains(&(j, i)) {
                continue;
            }
            if rng.next_f32() < config.extra_corridor_chance {
                let (x1, y1) = rooms[i].center;
                let (x2, y2) = rooms[j].center;
                let horizontal_first = rng.bool(0.5);
                let tiles = map.carve_l_corridor(x1, y1, x2, y2, horizontal_first);
                map.corridors.push(Corridor {
                    tiles,
                    from_room: i,
                    to_room: j,
                });
            }
        }
    }

    map.rooms = rooms;

    // Post-processing.
    map.place_stairs();
    map.place_doors();
    map.build_difficulty_map();

    map
}

// ===========================================================================
// A* pathfinding (used internally)
// ===========================================================================

/// A* pathfinding on the dungeon map.
///
/// Finds the shortest walkable path from `start` to `end`.
pub fn find_path(
    map: &DungeonMap,
    start: (usize, usize),
    end: (usize, usize),
) -> Option<Vec<(usize, usize)>> {
    if !map.get(start.0, start.1).is_walkable() || !map.get(end.0, end.1).is_walkable() {
        return None;
    }

    let heuristic = |a: (usize, usize), b: (usize, usize)| -> f32 {
        let dx = (a.0 as f32 - b.0 as f32).abs();
        let dy = (a.1 as f32 - b.1 as f32).abs();
        dx + dy // Manhattan distance.
    };

    #[derive(Debug, Clone, PartialEq)]
    struct Node {
        pos: (usize, usize),
        g: f32,
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
            other
                .f
                .partial_cmp(&self.f)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let mut open = BinaryHeap::new();
    let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut g_score: HashMap<(usize, usize), f32> = HashMap::new();

    g_score.insert(start, 0.0);
    open.push(Node {
        pos: start,
        g: 0.0,
        f: heuristic(start, end),
    });

    while let Some(current) = open.pop() {
        if current.pos == end {
            // Reconstruct path.
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

        for &(dx, dy) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
            let nx = current.pos.0 as i32 + dx;
            let ny = current.pos.1 as i32 + dy;
            if nx < 0 || ny < 0 {
                continue;
            }
            let neighbor = (nx as usize, ny as usize);
            if neighbor.0 >= map.width || neighbor.1 >= map.height {
                continue;
            }
            if !map.get(neighbor.0, neighbor.1).is_walkable() {
                continue;
            }

            let tentative_g = current_g + 1.0;
            let best_g = *g_score.get(&neighbor).unwrap_or(&f32::MAX);

            if tentative_g < best_g {
                came_from.insert(neighbor, current.pos);
                g_score.insert(neighbor, tentative_g);
                open.push(Node {
                    pos: neighbor,
                    g: tentative_g,
                    f: tentative_g + heuristic(neighbor, end),
                });
            }
        }
    }

    None // No path found.
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_room_creation() {
        let room = Room::new(0, 5, 5, 10, 8);
        assert_eq!(room.center, (10, 9));
        assert_eq!(room.area(), 80);
        assert_eq!(room.right(), 15);
        assert_eq!(room.bottom(), 13);
        assert!(room.contains(5, 5));
        assert!(room.contains(14, 12));
        assert!(!room.contains(15, 13));
    }

    #[test]
    fn test_room_overlap() {
        let a = Room::new(0, 5, 5, 10, 10);
        let b = Room::new(1, 10, 10, 10, 10);
        assert!(a.overlaps(&b, 0));

        let c = Room::new(2, 20, 20, 5, 5);
        assert!(!a.overlaps(&c, 0));
    }

    #[test]
    fn test_dungeon_map_basic() {
        let mut map = DungeonMap::new(10, 10);
        assert_eq!(map.get(0, 0), DungeonTile::Wall);
        map.set(5, 5, DungeonTile::Floor);
        assert_eq!(map.get(5, 5), DungeonTile::Floor);
    }

    #[test]
    fn test_bsp_generation() {
        let config = BSPConfig {
            width: 40,
            height: 30,
            min_room_size: 4,
            max_depth: 4,
            seed: 42,
            ..Default::default()
        };
        let map = generate_bsp(&config);

        assert!(!map.rooms.is_empty(), "BSP should generate at least one room");
        assert!(map.floor_count() > 0, "BSP should have floor tiles");
        assert!(
            map.player_start.is_some(),
            "BSP should place player start"
        );
    }

    #[test]
    fn test_bsp_deterministic() {
        let config = BSPConfig::default();
        let map1 = generate_bsp(&config);
        let map2 = generate_bsp(&config);
        assert_eq!(map1.tiles, map2.tiles);
        assert_eq!(map1.rooms.len(), map2.rooms.len());
    }

    #[test]
    fn test_cave_generation() {
        let config = CaveConfig {
            width: 40,
            height: 30,
            seed: 42,
            ..Default::default()
        };
        let map = generate_cave(&config);

        assert!(map.floor_count() > 0, "Cave should have floor tiles");
        // The cave should be connected (only one region after cleanup).
    }

    #[test]
    fn test_cave_has_entrance_exit() {
        let config = CaveConfig {
            width: 40,
            height: 30,
            seed: 123,
            ..Default::default()
        };
        let map = generate_cave(&config);

        assert!(map.player_start.is_some());
        assert!(map.exit.is_some());
        if let (Some(start), Some(exit)) = (map.player_start, map.exit) {
            assert_ne!(start, exit, "Start and exit should be different");
        }
    }

    #[test]
    fn test_drunkard_generation() {
        let config = DrunkardConfig {
            width: 40,
            height: 30,
            target_floor_percent: 0.3,
            seed: 42,
            ..Default::default()
        };
        let map = generate_drunkard(&config);

        let total = config.width * config.height;
        let floor = map.floor_count();
        let ratio = floor as f32 / total as f32;
        // Should be close to the target (within reasonable margin).
        assert!(
            ratio > 0.1,
            "Drunkard should carve at least some floor, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_room_placement() {
        let config = RoomPlacementConfig {
            width: 60,
            height: 40,
            room_count: 8,
            seed: 42,
            ..Default::default()
        };
        let map = generate_room_placement(&config);

        assert!(
            map.rooms.len() >= 2,
            "Should place at least 2 rooms, got {}",
            map.rooms.len()
        );
        assert!(
            !map.corridors.is_empty(),
            "Should have corridors connecting rooms"
        );
    }

    #[test]
    fn test_room_placement_connectivity() {
        let config = RoomPlacementConfig {
            width: 80,
            height: 50,
            room_count: 10,
            seed: 42,
            ..Default::default()
        };
        let map = generate_room_placement(&config);

        // All rooms should be reachable from the first room.
        if map.rooms.len() >= 2 {
            let start = map.rooms[0].center;
            for room in &map.rooms[1..] {
                let path = find_path(&map, start, room.center);
                assert!(
                    path.is_some(),
                    "Room {} should be reachable from room 0",
                    room.id
                );
            }
        }
    }

    #[test]
    fn test_difficulty_map() {
        let config = BSPConfig {
            width: 40,
            height: 30,
            seed: 42,
            ..Default::default()
        };
        let map = generate_bsp(&config);

        if let Some(start) = map.player_start {
            let idx = start.1 * map.width + start.0;
            assert_eq!(
                map.difficulty_map[idx], 0,
                "Difficulty at player start should be 0"
            );
        }
    }

    #[test]
    fn test_pathfinding() {
        let config = RoomPlacementConfig {
            width: 40,
            height: 30,
            room_count: 5,
            seed: 42,
            ..Default::default()
        };
        let map = generate_room_placement(&config);

        if let (Some(start), Some(exit)) = (map.player_start, map.exit) {
            let path = find_path(&map, start, exit);
            assert!(path.is_some(), "Should find path from start to exit");
            let path = path.unwrap();
            assert!(path.len() >= 2, "Path should have at least 2 points");
            assert_eq!(path[0], start);
            assert_eq!(path[path.len() - 1], exit);
        }
    }

    #[test]
    fn test_ascii_rendering() {
        let config = BSPConfig {
            width: 20,
            height: 15,
            min_room_size: 3,
            max_depth: 3,
            seed: 42,
            ..Default::default()
        };
        let map = generate_bsp(&config);
        let ascii = map.to_ascii();

        assert!(!ascii.is_empty());
        assert!(ascii.contains('#'), "ASCII should contain wall characters");
        assert!(ascii.contains('.'), "ASCII should contain floor characters");
    }

    #[test]
    fn test_tile_properties() {
        assert!(DungeonTile::Floor.is_walkable());
        assert!(DungeonTile::Door.is_walkable());
        assert!(DungeonTile::StairsUp.is_walkable());
        assert!(!DungeonTile::Wall.is_walkable());
        assert!(DungeonTile::Wall.is_solid());
        assert!(!DungeonTile::Floor.is_solid());
    }
}
