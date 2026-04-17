// engine/procgen/src/dungeon_v2.rs
//
// Enhanced dungeon generator: BSP + cellular automata hybrid,
// room decoration, enemy/loot/trap placement, difficulty scaling,
// themed rooms, and corridor carving with multiple strategies.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Simple RNG (xorshift64)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }

    pub fn range(&mut self, min: i32, max: i32) -> i32 {
        if min >= max { return min; }
        min + (self.next_u64() % (max - min) as u64) as i32
    }

    pub fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }

    pub fn chance(&mut self, probability: f32) -> bool {
        self.next_f32() < probability
    }

    pub fn pick<T: Clone>(&mut self, items: &[T]) -> T {
        let idx = (self.next_u64() % items.len() as u64) as usize;
        items[idx].clone()
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Tile types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tile {
    Wall,
    Floor,
    Corridor,
    Door,
    StairsUp,
    StairsDown,
    Water,
    Lava,
    Pit,
    SecretWall,
    Decoration,
}

impl Tile {
    pub fn is_walkable(&self) -> bool {
        matches!(self, Self::Floor | Self::Corridor | Self::Door | Self::StairsUp | Self::StairsDown | Self::Decoration)
    }

    pub fn is_solid(&self) -> bool {
        matches!(self, Self::Wall | Self::SecretWall)
    }
}

// ---------------------------------------------------------------------------
// Room types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomTheme {
    Normal,
    Treasure,
    Boss,
    Shop,
    Library,
    Armory,
    Prison,
    Garden,
    Shrine,
    Cave,
    Flooded,
    Trap,
    Secret,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomShape {
    Rectangular,
    Circular,
    LShaped,
    Cross,
    Irregular,
}

#[derive(Debug, Clone)]
pub struct Room {
    pub id: u32,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub theme: RoomTheme,
    pub shape: RoomShape,
    pub difficulty: f32,
    pub connected_to: Vec<u32>,
    pub has_entrance: bool,
    pub has_exit: bool,
    pub decorations: Vec<Decoration>,
    pub enemies: Vec<EnemyPlacement>,
    pub loot: Vec<LootPlacement>,
    pub traps: Vec<TrapPlacement>,
}

impl Room {
    pub fn center(&self) -> (i32, i32) {
        (self.x + self.width / 2, self.y + self.height / 2)
    }

    pub fn area(&self) -> i32 { self.width * self.height }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    pub fn intersects(&self, other: &Room, margin: i32) -> bool {
        self.x - margin < other.x + other.width
            && self.x + self.width + margin > other.x
            && self.y - margin < other.y + other.height
            && self.y + self.height + margin > other.y
    }

    pub fn distance_to(&self, other: &Room) -> f32 {
        let (cx1, cy1) = self.center();
        let (cx2, cy2) = other.center();
        let dx = (cx2 - cx1) as f32;
        let dy = (cy2 - cy1) as f32;
        (dx * dx + dy * dy).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Placement types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Decoration {
    pub x: i32,
    pub y: i32,
    pub decoration_type: DecorationType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecorationType {
    Pillar,
    Statue,
    Fountain,
    Torch,
    Bookshelf,
    Table,
    Chair,
    Barrel,
    Crate,
    WeaponRack,
    Banner,
    Rubble,
    Vines,
    Bones,
    Cobweb,
    Altar,
    Well,
    CrystalFormation,
}

#[derive(Debug, Clone)]
pub struct EnemyPlacement {
    pub x: i32,
    pub y: i32,
    pub enemy_type: EnemyType,
    pub level: u32,
    pub is_boss: bool,
    pub patrol_path: Vec<(i32, i32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnemyType {
    Skeleton,
    Zombie,
    Goblin,
    Orc,
    Spider,
    Bat,
    Ghost,
    Slime,
    Mimic,
    Dragon,
    Cultist,
    Golem,
}

#[derive(Debug, Clone)]
pub struct LootPlacement {
    pub x: i32,
    pub y: i32,
    pub loot_type: LootType,
    pub rarity: LootRarity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LootType {
    Chest,
    GoldPile,
    Potion,
    Scroll,
    Weapon,
    Armor,
    Ring,
    Key,
    QuestItem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LootRarity {
    Common,
    Uncommon,
    Rare,
    Epic,
    Legendary,
}

#[derive(Debug, Clone)]
pub struct TrapPlacement {
    pub x: i32,
    pub y: i32,
    pub trap_type: TrapType,
    pub armed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrapType {
    Spikes,
    ArrowTrap,
    PitTrap,
    PoisonGas,
    FireJet,
    BoulderTrap,
    TeleportTrap,
    AlarmTrap,
}

// ---------------------------------------------------------------------------
// BSP tree
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BspNode {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    left: Option<Box<BspNode>>,
    right: Option<Box<BspNode>>,
    room: Option<Room>,
}

impl BspNode {
    fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, width: w, height: h, left: None, right: None, room: None }
    }

    fn split(&mut self, rng: &mut Rng, min_size: i32, depth: i32, max_depth: i32) -> bool {
        if depth >= max_depth { return false; }
        if self.left.is_some() || self.right.is_some() { return false; }

        let can_split_h = self.width >= min_size * 2;
        let can_split_v = self.height >= min_size * 2;
        if !can_split_h && !can_split_v { return false; }

        let split_horizontal = if can_split_h && can_split_v {
            rng.chance(0.5)
        } else {
            can_split_h
        };

        if split_horizontal {
            let split = rng.range(min_size, self.width - min_size + 1);
            self.left = Some(Box::new(BspNode::new(self.x, self.y, split, self.height)));
            self.right = Some(Box::new(BspNode::new(self.x + split, self.y, self.width - split, self.height)));
        } else {
            let split = rng.range(min_size, self.height - min_size + 1);
            self.left = Some(Box::new(BspNode::new(self.x, self.y, self.width, split)));
            self.right = Some(Box::new(BspNode::new(self.x, self.y + split, self.width, self.height - split)));
        }

        if let Some(ref mut left) = self.left {
            left.split(rng, min_size, depth + 1, max_depth);
        }
        if let Some(ref mut right) = self.right {
            right.split(rng, min_size, depth + 1, max_depth);
        }

        true
    }

    fn create_rooms(&mut self, rng: &mut Rng, room_id: &mut u32, min_room: i32) {
        if self.left.is_none() && self.right.is_none() {
            // Leaf node: create a room.
            let room_w = rng.range(min_room, self.width - 1).max(min_room);
            let room_h = rng.range(min_room, self.height - 1).max(min_room);
            let room_x = self.x + rng.range(1, (self.width - room_w).max(1) + 1);
            let room_y = self.y + rng.range(1, (self.height - room_h).max(1) + 1);

            let id = *room_id;
            *room_id += 1;

            self.room = Some(Room {
                id,
                x: room_x,
                y: room_y,
                width: room_w,
                height: room_h,
                theme: RoomTheme::Normal,
                shape: RoomShape::Rectangular,
                difficulty: 0.0,
                connected_to: Vec::new(),
                has_entrance: false,
                has_exit: false,
                decorations: Vec::new(),
                enemies: Vec::new(),
                loot: Vec::new(),
                traps: Vec::new(),
            });
        } else {
            if let Some(ref mut left) = self.left {
                left.create_rooms(rng, room_id, min_room);
            }
            if let Some(ref mut right) = self.right {
                right.create_rooms(rng, room_id, min_room);
            }
        }
    }

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

    fn get_room_center(&self) -> Option<(i32, i32)> {
        if let Some(ref room) = self.room {
            return Some(room.center());
        }
        if let Some(ref left) = self.left {
            if let Some(c) = left.get_room_center() { return Some(c); }
        }
        if let Some(ref right) = self.right {
            if let Some(c) = right.get_room_center() { return Some(c); }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Dungeon generator config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DungeonConfig {
    pub width: i32,
    pub height: i32,
    pub seed: u64,
    pub min_room_size: i32,
    pub max_bsp_depth: i32,
    pub corridor_width: i32,
    pub room_margin: i32,
    pub difficulty_level: u32,
    pub enemy_density: f32,
    pub loot_density: f32,
    pub trap_density: f32,
    pub decoration_density: f32,
    pub cave_ratio: f32,
    pub cave_iterations: u32,
    pub cave_birth: u32,
    pub cave_death: u32,
    pub secret_room_chance: f32,
    pub boss_room: bool,
    pub shop_room: bool,
}

impl Default for DungeonConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 60,
            seed: 42,
            min_room_size: 5,
            max_bsp_depth: 5,
            corridor_width: 1,
            room_margin: 1,
            difficulty_level: 1,
            enemy_density: 0.3,
            loot_density: 0.15,
            trap_density: 0.1,
            decoration_density: 0.2,
            cave_ratio: 0.0,
            cave_iterations: 4,
            cave_birth: 4,
            cave_death: 3,
            secret_room_chance: 0.1,
            boss_room: true,
            shop_room: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Generated dungeon
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GeneratedDungeon {
    pub tiles: Vec<Vec<Tile>>,
    pub width: i32,
    pub height: i32,
    pub rooms: Vec<Room>,
    pub corridors: Vec<(i32, i32)>,
    pub entrance: (i32, i32),
    pub exit: (i32, i32),
    pub seed: u64,
    pub difficulty: u32,
}

impl GeneratedDungeon {
    pub fn get_tile(&self, x: i32, y: i32) -> Tile {
        if x < 0 || y < 0 || x >= self.width || y >= self.height {
            return Tile::Wall;
        }
        self.tiles[y as usize][x as usize]
    }

    pub fn set_tile(&mut self, x: i32, y: i32, tile: Tile) {
        if x >= 0 && y >= 0 && x < self.width && y < self.height {
            self.tiles[y as usize][x as usize] = tile;
        }
    }

    pub fn room_at(&self, x: i32, y: i32) -> Option<&Room> {
        self.rooms.iter().find(|r| r.contains(x, y))
    }

    pub fn total_enemies(&self) -> usize {
        self.rooms.iter().map(|r| r.enemies.len()).sum()
    }

    pub fn total_loot(&self) -> usize {
        self.rooms.iter().map(|r| r.loot.len()).sum()
    }

    pub fn total_traps(&self) -> usize {
        self.rooms.iter().map(|r| r.traps.len()).sum()
    }

    pub fn walkable_tiles(&self) -> usize {
        self.tiles.iter()
            .flat_map(|row| row.iter())
            .filter(|t| t.is_walkable())
            .count()
    }
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// Generate a complete dungeon.
pub fn generate_dungeon(config: &DungeonConfig) -> GeneratedDungeon {
    let mut rng = Rng::new(config.seed);
    let w = config.width;
    let h = config.height;

    // Initialize map with walls.
    let mut tiles = vec![vec![Tile::Wall; w as usize]; h as usize];

    // BSP split.
    let mut bsp = BspNode::new(0, 0, w, h);
    bsp.split(&mut rng, config.min_room_size + 2, 0, config.max_bsp_depth);

    // Create rooms.
    let mut room_id = 0u32;
    bsp.create_rooms(&mut rng, &mut room_id, config.min_room_size);

    let mut rooms = Vec::new();
    bsp.collect_rooms(&mut rooms);

    // Carve rooms into the tile map.
    for room in &rooms {
        carve_room(&mut tiles, room, w, h);
    }

    // Connect rooms with corridors.
    let mut corridors = Vec::new();
    for i in 0..rooms.len().saturating_sub(1) {
        let (cx1, cy1) = rooms[i].center();
        let (cx2, cy2) = rooms[i + 1].center();
        carve_corridor(&mut tiles, cx1, cy1, cx2, cy2, &mut rng, &mut corridors, w, h);
        rooms[i].connected_to.push(rooms[i + 1].id);
        rooms[i + 1].connected_to.push(rooms[i].id);
    }

    // Apply cellular automata for cave sections.
    if config.cave_ratio > 0.0 {
        apply_cave_generation(&mut tiles, &mut rng, config, w, h);
    }

    // Place doors at corridor/room boundaries.
    place_doors(&mut tiles, &rooms, w, h, &mut rng);

    // Assign themes to rooms.
    assign_themes(&mut rooms, &mut rng, config);

    // Place entrance and exit.
    let entrance = rooms.first().map(|r| r.center()).unwrap_or((1, 1));
    let exit = rooms.last().map(|r| r.center()).unwrap_or((w - 2, h - 2));
    if !rooms.is_empty() {
        rooms.first_mut().unwrap().has_entrance = true;
        rooms.last_mut().unwrap().has_exit = true;
    }
    tiles[entrance.1 as usize][entrance.0 as usize] = Tile::StairsUp;
    tiles[exit.1 as usize][exit.0 as usize] = Tile::StairsDown;

    // Place content.
    for room in &mut rooms {
        place_decorations(room, &mut rng, config);
        place_enemies(room, &mut rng, config);
        place_loot(room, &mut rng, config);
        place_traps(room, &mut rng, config);
    }

    // Assign difficulty to rooms based on distance from entrance.
    let entrance_room = rooms.first().map(|r| r.id);
    for (i, room) in rooms.iter_mut().enumerate() {
        room.difficulty = (i as f32 / rooms.len().max(1) as f32) * config.difficulty_level as f32;
    }

    GeneratedDungeon {
        tiles,
        width: w,
        height: h,
        rooms,
        corridors,
        entrance,
        exit,
        seed: config.seed,
        difficulty: config.difficulty_level,
    }
}

fn carve_room(tiles: &mut [Vec<Tile>], room: &Room, map_w: i32, map_h: i32) {
    for y in room.y..room.y + room.height {
        for x in room.x..room.x + room.width {
            if x > 0 && x < map_w - 1 && y > 0 && y < map_h - 1 {
                tiles[y as usize][x as usize] = Tile::Floor;
            }
        }
    }
}

fn carve_corridor(
    tiles: &mut [Vec<Tile>],
    x1: i32, y1: i32, x2: i32, y2: i32,
    rng: &mut Rng,
    corridors: &mut Vec<(i32, i32)>,
    map_w: i32, map_h: i32,
) {
    let (mut cx, mut cy) = (x1, y1);

    // Random L-shaped corridor.
    let horizontal_first = rng.chance(0.5);

    if horizontal_first {
        // Horizontal then vertical.
        while cx != x2 {
            if cx > 0 && cx < map_w - 1 && cy > 0 && cy < map_h - 1 {
                if tiles[cy as usize][cx as usize] == Tile::Wall {
                    tiles[cy as usize][cx as usize] = Tile::Corridor;
                    corridors.push((cx, cy));
                }
            }
            cx += if x2 > cx { 1 } else { -1 };
        }
        while cy != y2 {
            if cx > 0 && cx < map_w - 1 && cy > 0 && cy < map_h - 1 {
                if tiles[cy as usize][cx as usize] == Tile::Wall {
                    tiles[cy as usize][cx as usize] = Tile::Corridor;
                    corridors.push((cx, cy));
                }
            }
            cy += if y2 > cy { 1 } else { -1 };
        }
    } else {
        // Vertical then horizontal.
        while cy != y2 {
            if cx > 0 && cx < map_w - 1 && cy > 0 && cy < map_h - 1 {
                if tiles[cy as usize][cx as usize] == Tile::Wall {
                    tiles[cy as usize][cx as usize] = Tile::Corridor;
                    corridors.push((cx, cy));
                }
            }
            cy += if y2 > cy { 1 } else { -1 };
        }
        while cx != x2 {
            if cx > 0 && cx < map_w - 1 && cy > 0 && cy < map_h - 1 {
                if tiles[cy as usize][cx as usize] == Tile::Wall {
                    tiles[cy as usize][cx as usize] = Tile::Corridor;
                    corridors.push((cx, cy));
                }
            }
            cx += if x2 > cx { 1 } else { -1 };
        }
    }
}

fn apply_cave_generation(tiles: &mut [Vec<Tile>], rng: &mut Rng, config: &DungeonConfig, w: i32, h: i32) {
    // Only affect wall tiles in certain regions.
    let cave_area_y = (h as f32 * (1.0 - config.cave_ratio)) as i32;

    // Initialize cave region with random fill.
    for y in cave_area_y..h - 1 {
        for x in 1..w - 1 {
            if tiles[y as usize][x as usize] == Tile::Wall {
                if rng.chance(0.45) {
                    tiles[y as usize][x as usize] = Tile::Floor;
                }
            }
        }
    }

    // Run cellular automata.
    for _ in 0..config.cave_iterations {
        let mut new_tiles = tiles.to_vec();
        for y in cave_area_y..h - 1 {
            for x in 1..w - 1 {
                let neighbors = count_wall_neighbors(tiles, x, y, w, h);
                if tiles[y as usize][x as usize] == Tile::Wall {
                    if neighbors < config.cave_death {
                        new_tiles[y as usize][x as usize] = Tile::Floor;
                    }
                } else if tiles[y as usize][x as usize] != Tile::Corridor {
                    if neighbors >= config.cave_birth {
                        new_tiles[y as usize][x as usize] = Tile::Wall;
                    }
                }
            }
        }
        for y in cave_area_y..h - 1 {
            for x in 1..w - 1 {
                tiles[y as usize][x as usize] = new_tiles[y as usize][x as usize];
            }
        }
    }
}

fn count_wall_neighbors(tiles: &[Vec<Tile>], x: i32, y: i32, w: i32, h: i32) -> u32 {
    let mut count = 0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || nx >= w || ny < 0 || ny >= h {
                count += 1;
            } else if tiles[ny as usize][nx as usize].is_solid() {
                count += 1;
            }
        }
    }
    count
}

fn place_doors(tiles: &mut [Vec<Tile>], rooms: &[Room], w: i32, h: i32, rng: &mut Rng) {
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if tiles[y as usize][x as usize] != Tile::Corridor { continue; }

            // Check if this corridor tile is adjacent to a room floor.
            let is_door_candidate =
                (tiles[(y - 1) as usize][x as usize] == Tile::Floor
                    && tiles[(y + 1) as usize][x as usize] == Tile::Floor
                    && tiles[y as usize][(x - 1) as usize].is_solid()
                    && tiles[y as usize][(x + 1) as usize].is_solid())
                || (tiles[y as usize][(x - 1) as usize] == Tile::Floor
                    && tiles[y as usize][(x + 1) as usize] == Tile::Floor
                    && tiles[(y - 1) as usize][x as usize].is_solid()
                    && tiles[(y + 1) as usize][x as usize].is_solid());

            if is_door_candidate && rng.chance(0.6) {
                tiles[y as usize][x as usize] = Tile::Door;
            }
        }
    }
}

fn assign_themes(rooms: &mut Vec<Room>, rng: &mut Rng, config: &DungeonConfig) {
    if rooms.is_empty() { return; }

    // Last room is boss room.
    if config.boss_room && rooms.len() > 2 {
        rooms.last_mut().unwrap().theme = RoomTheme::Boss;
    }

    // Random shop room.
    if config.shop_room && rooms.len() > 3 {
        let idx = rng.range(1, rooms.len() as i32 - 1) as usize;
        rooms[idx].theme = RoomTheme::Shop;
    }

    // Assign themed rooms randomly.
    let themes = [
        RoomTheme::Library, RoomTheme::Armory, RoomTheme::Prison,
        RoomTheme::Shrine, RoomTheme::Cave, RoomTheme::Garden,
    ];
    for room in rooms.iter_mut() {
        if room.theme == RoomTheme::Normal && rng.chance(0.2) {
            room.theme = rng.pick(&themes);
        }
    }

    // Secret rooms.
    if config.secret_room_chance > 0.0 {
        for room in rooms.iter_mut() {
            if room.theme == RoomTheme::Normal && rng.chance(config.secret_room_chance) {
                room.theme = RoomTheme::Secret;
            }
        }
    }
}

fn place_decorations(room: &mut Room, rng: &mut Rng, config: &DungeonConfig) {
    if config.decoration_density <= 0.0 { return; }

    let themed_decos: Vec<DecorationType> = match room.theme {
        RoomTheme::Library => vec![DecorationType::Bookshelf, DecorationType::Table, DecorationType::Cobweb],
        RoomTheme::Armory => vec![DecorationType::WeaponRack, DecorationType::Barrel, DecorationType::Banner],
        RoomTheme::Shrine => vec![DecorationType::Altar, DecorationType::Pillar, DecorationType::Torch],
        RoomTheme::Cave => vec![DecorationType::CrystalFormation, DecorationType::Rubble, DecorationType::Bones],
        RoomTheme::Garden => vec![DecorationType::Vines, DecorationType::Fountain, DecorationType::Well],
        _ => vec![DecorationType::Torch, DecorationType::Barrel, DecorationType::Crate, DecorationType::Pillar],
    };

    let count = ((room.area() as f32 * config.decoration_density) as i32).max(1);
    for _ in 0..count {
        let dx = rng.range(1, room.width - 1);
        let dy = rng.range(1, room.height - 1);
        room.decorations.push(Decoration {
            x: room.x + dx,
            y: room.y + dy,
            decoration_type: rng.pick(&themed_decos),
        });
    }
}

fn place_enemies(room: &mut Room, rng: &mut Rng, config: &DungeonConfig) {
    if config.enemy_density <= 0.0 { return; }
    if room.theme == RoomTheme::Shop { return; }
    if room.has_entrance { return; }

    let enemy_types: Vec<EnemyType> = match room.difficulty as u32 {
        0 => vec![EnemyType::Bat, EnemyType::Slime, EnemyType::Goblin],
        1 => vec![EnemyType::Skeleton, EnemyType::Zombie, EnemyType::Spider],
        2 => vec![EnemyType::Orc, EnemyType::Cultist, EnemyType::Ghost],
        _ => vec![EnemyType::Golem, EnemyType::Dragon],
    };

    let count = ((room.area() as f32 * config.enemy_density * 0.1) as i32).max(1);
    let is_boss = room.theme == RoomTheme::Boss;

    for _ in 0..count {
        let dx = rng.range(1, room.width - 1);
        let dy = rng.range(1, room.height - 1);
        room.enemies.push(EnemyPlacement {
            x: room.x + dx,
            y: room.y + dy,
            enemy_type: if is_boss { EnemyType::Dragon } else { rng.pick(&enemy_types) },
            level: config.difficulty_level + (room.difficulty as u32),
            is_boss,
            patrol_path: Vec::new(),
        });
    }
}

fn place_loot(room: &mut Room, rng: &mut Rng, config: &DungeonConfig) {
    if config.loot_density <= 0.0 { return; }

    let count = if room.theme == RoomTheme::Treasure {
        3
    } else {
        ((room.area() as f32 * config.loot_density * 0.05) as i32).max(0)
    };

    for _ in 0..count {
        let dx = rng.range(1, room.width - 1);
        let dy = rng.range(1, room.height - 1);

        let rarity = if rng.chance(0.05) {
            LootRarity::Legendary
        } else if rng.chance(0.1) {
            LootRarity::Epic
        } else if rng.chance(0.2) {
            LootRarity::Rare
        } else if rng.chance(0.3) {
            LootRarity::Uncommon
        } else {
            LootRarity::Common
        };

        let loot_types = [LootType::Chest, LootType::GoldPile, LootType::Potion, LootType::Scroll];
        room.loot.push(LootPlacement {
            x: room.x + dx,
            y: room.y + dy,
            loot_type: rng.pick(&loot_types),
            rarity,
        });
    }
}

fn place_traps(room: &mut Room, rng: &mut Rng, config: &DungeonConfig) {
    if config.trap_density <= 0.0 { return; }
    if room.theme == RoomTheme::Shop { return; }
    if room.has_entrance { return; }

    let trap_chance = config.trap_density * (room.difficulty + 1.0) * 0.1;
    if !rng.chance(trap_chance) { return; }

    let trap_types = [TrapType::Spikes, TrapType::ArrowTrap, TrapType::PoisonGas, TrapType::FireJet];
    let dx = rng.range(1, room.width - 1);
    let dy = rng.range(1, room.height - 1);
    room.traps.push(TrapPlacement {
        x: room.x + dx,
        y: room.y + dy,
        trap_type: rng.pick(&trap_types),
        armed: true,
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_default() {
        let dungeon = generate_dungeon(&DungeonConfig::default());
        assert!(dungeon.rooms.len() > 0);
        assert!(dungeon.walkable_tiles() > 0);
        assert_ne!(dungeon.entrance, dungeon.exit);
    }

    #[test]
    fn test_deterministic() {
        let d1 = generate_dungeon(&DungeonConfig { seed: 12345, ..Default::default() });
        let d2 = generate_dungeon(&DungeonConfig { seed: 12345, ..Default::default() });
        assert_eq!(d1.rooms.len(), d2.rooms.len());
        assert_eq!(d1.entrance, d2.entrance);
    }

    #[test]
    fn test_different_seeds() {
        let d1 = generate_dungeon(&DungeonConfig { seed: 1, ..Default::default() });
        let d2 = generate_dungeon(&DungeonConfig { seed: 999, ..Default::default() });
        // Very unlikely to be identical.
        assert!(d1.rooms.len() != d2.rooms.len() || d1.entrance != d2.entrance);
    }

    #[test]
    fn test_with_caves() {
        let config = DungeonConfig { cave_ratio: 0.3, ..Default::default() };
        let dungeon = generate_dungeon(&config);
        assert!(dungeon.walkable_tiles() > 0);
    }

    #[test]
    fn test_enemy_placement() {
        let config = DungeonConfig { enemy_density: 0.5, ..Default::default() };
        let dungeon = generate_dungeon(&config);
        assert!(dungeon.total_enemies() > 0);
    }

    #[test]
    fn test_loot_placement() {
        let config = DungeonConfig { loot_density: 0.5, ..Default::default() };
        let dungeon = generate_dungeon(&config);
        assert!(dungeon.total_loot() > 0);
    }
}
