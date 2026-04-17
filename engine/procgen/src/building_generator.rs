//! Building generation: floor plan generation, room classification, door/window
//! placement, interior decoration, exterior style, and multi-story buildings.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomType { Living, Kitchen, Bedroom, Bathroom, Hallway, Stairs, Storage, Office, Shop, Tavern, Workshop, Temple, Throne, Dungeon, Library, Armory }
impl RoomType {
    pub fn min_area(&self) -> f32 { match self { Self::Bathroom | Self::Storage => 4.0, Self::Hallway | Self::Stairs => 3.0, Self::Kitchen | Self::Office => 8.0, Self::Bedroom => 9.0, Self::Living | Self::Shop => 12.0, Self::Tavern | Self::Workshop => 16.0, Self::Library | Self::Armory => 20.0, Self::Temple | Self::Throne => 30.0, Self::Dungeon => 10.0 } }
    pub fn max_area(&self) -> f32 { self.min_area() * 3.0 }
    pub fn needs_window(&self) -> bool { !matches!(self, Self::Storage | Self::Dungeon | Self::Stairs | Self::Hallway) }
    pub fn needs_exterior_wall(&self) -> bool { self.needs_window() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExteriorStyle { Medieval, Tudor, Stone, Wooden, Brick, Plaster, Castle, Elven, Dwarven, Modern }
impl ExteriorStyle {
    pub fn wall_material(&self) -> &'static str { match self { Self::Medieval | Self::Tudor => "timber_frame", Self::Stone | Self::Castle | Self::Dwarven => "stone", Self::Wooden | Self::Elven => "wood", Self::Brick | Self::Modern => "brick", Self::Plaster => "plaster" } }
    pub fn roof_material(&self) -> &'static str { match self { Self::Castle | Self::Stone => "stone_slate", Self::Modern => "tile", _ => "thatch" } }
    pub fn floor_height(&self) -> f32 { match self { Self::Castle | Self::Temple => 4.0, Self::Dwarven => 3.0, _ => 3.0 } }
}

#[derive(Debug, Clone)]
pub struct Room { pub id: u32, pub room_type: RoomType, pub x: f32, pub z: f32, pub width: f32, pub depth: f32, pub floor: u32, pub doors: Vec<DoorPlacement>, pub windows: Vec<WindowPlacement>, pub furniture: Vec<FurniturePlacement> }

impl Room {
    pub fn new(id: u32, rt: RoomType, x: f32, z: f32, w: f32, d: f32, floor: u32) -> Self {
        Self { id, room_type: rt, x, z, width: w, depth: d, floor, doors: Vec::new(), windows: Vec::new(), furniture: Vec::new() }
    }
    pub fn area(&self) -> f32 { self.width * self.depth }
    pub fn center(&self) -> [f32; 2] { [self.x + self.width * 0.5, self.z + self.depth * 0.5] }
    pub fn contains(&self, px: f32, pz: f32) -> bool { px >= self.x && px <= self.x + self.width && pz >= self.z && pz <= self.z + self.depth }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallSide { North, South, East, West }

#[derive(Debug, Clone)]
pub struct DoorPlacement { pub wall: WallSide, pub position: f32, pub width: f32, pub height: f32, pub is_exterior: bool }
#[derive(Debug, Clone)]
pub struct WindowPlacement { pub wall: WallSide, pub position: f32, pub width: f32, pub height: f32, pub sill_height: f32, pub style: WindowStyle }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowStyle { Rectangular, Arched, Round, Stained, Arrow }
#[derive(Debug, Clone)]
pub struct FurniturePlacement { pub furniture_type: String, pub x: f32, pub z: f32, pub rotation: f32, pub scale: f32 }

#[derive(Debug, Clone)]
pub struct FloorPlan { pub floor: u32, pub rooms: Vec<Room>, pub width: f32, pub depth: f32 }
impl FloorPlan {
    pub fn new(floor: u32, width: f32, depth: f32) -> Self { Self { floor, rooms: Vec::new(), width, depth } }
    pub fn add_room(&mut self, room: Room) { self.rooms.push(room); }
    pub fn room_count(&self) -> usize { self.rooms.len() }
    pub fn total_area(&self) -> f32 { self.rooms.iter().map(|r| r.area()).sum() }
}

#[derive(Debug, Clone)]
pub struct RoofConfig { pub roof_type: RoofType, pub pitch: f32, pub overhang: f32, pub material: String }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoofType { Flat, Gable, Hip, Shed, Mansard, Dome, Pyramidal }

#[derive(Debug, Clone)]
pub struct BuildingConfig {
    pub style: ExteriorStyle, pub floors: u32, pub width: f32, pub depth: f32, pub roof: RoofConfig,
    pub seed: u64, pub room_types: Vec<RoomType>, pub has_basement: bool, pub has_chimney: bool,
    pub has_porch: bool, pub door_style: String, pub window_style: WindowStyle,
}

impl Default for BuildingConfig {
    fn default() -> Self {
        Self {
            style: ExteriorStyle::Medieval, floors: 2, width: 10.0, depth: 8.0,
            roof: RoofConfig { roof_type: RoofType::Gable, pitch: 35.0, overhang: 0.5, material: "thatch".to_string() },
            seed: 42, room_types: vec![RoomType::Living, RoomType::Kitchen, RoomType::Bedroom, RoomType::Bathroom],
            has_basement: false, has_chimney: true, has_porch: false,
            door_style: "wooden".to_string(), window_style: WindowStyle::Rectangular,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedBuilding { pub config: BuildingConfig, pub floors: Vec<FloorPlan>, pub total_rooms: u32, pub total_area: f32, pub exterior_walls: Vec<WallSegment>, pub height: f32 }

#[derive(Debug, Clone)]
pub struct WallSegment { pub start: [f32; 2], pub end: [f32; 2], pub height: f32, pub material: String, pub has_window: bool, pub has_door: bool }

fn hash_val(a: u32, b: u32, seed: u64) -> f32 {
    let mut h = seed.wrapping_mul(6364136223846793005).wrapping_add(a as u64);
    h ^= b as u64; h = h.wrapping_mul(6364136223846793005);
    ((h >> 33) as f32) / (u32::MAX as f32)
}

pub fn generate_building(config: &BuildingConfig) -> GeneratedBuilding {
    let mut floors = Vec::new();
    let mut total_rooms = 0u32;
    let mut total_area = 0.0f32;
    let fh = config.style.floor_height();

    for floor_idx in 0..config.floors {
        let mut plan = FloorPlan::new(floor_idx, config.width, config.depth);
        let rooms_on_floor = if floor_idx == 0 { 3.min(config.room_types.len()) } else { 2.min(config.room_types.len()) };
        let room_width = config.width / rooms_on_floor as f32;

        for i in 0..rooms_on_floor {
            let rt_idx = (floor_idx as usize * rooms_on_floor + i) % config.room_types.len();
            let rt = config.room_types[rt_idx];
            let rx = i as f32 * room_width;
            let mut room = Room::new(total_rooms, rt, rx, 0.0, room_width, config.depth, floor_idx);

            // Add door
            if i > 0 { room.doors.push(DoorPlacement { wall: WallSide::West, position: config.depth * 0.5, width: 0.9, height: 2.1, is_exterior: false }); }
            if floor_idx == 0 && i == 0 { room.doors.push(DoorPlacement { wall: WallSide::South, position: room_width * 0.5, width: 1.0, height: 2.2, is_exterior: true }); }

            // Add windows
            if rt.needs_window() {
                room.windows.push(WindowPlacement { wall: WallSide::North, position: room_width * 0.5, width: 1.0, height: 1.2, sill_height: 0.9, style: config.window_style });
            }

            // Furniture
            let furn = match rt {
                RoomType::Bedroom => vec!["bed", "wardrobe", "nightstand"],
                RoomType::Kitchen => vec!["table", "stove", "cabinet"],
                RoomType::Living => vec!["sofa", "table", "bookshelf", "fireplace"],
                RoomType::Bathroom => vec!["tub", "basin"],
                _ => vec![],
            };
            for (fi, fname) in furn.iter().enumerate() {
                let fx = hash_val(total_rooms, fi as u32, config.seed) * (room_width - 1.0) + 0.5;
                let fz = hash_val(fi as u32, total_rooms, config.seed + 1) * (config.depth - 1.0) + 0.5;
                room.furniture.push(FurniturePlacement { furniture_type: fname.to_string(), x: fx, z: fz, rotation: hash_val(total_rooms, fi as u32 + 100, config.seed) * 360.0, scale: 1.0 });
            }

            total_area += room.area();
            total_rooms += 1;
            plan.add_room(room);
        }
        floors.push(plan);
    }

    let height = config.floors as f32 * fh + if config.roof.roof_type == RoofType::Flat { 0.3 } else { config.width * 0.5 * (config.roof.pitch.to_radians().tan()) };

    let mat = config.style.wall_material().to_string();
    let walls = vec![
        WallSegment { start: [0.0, 0.0], end: [config.width, 0.0], height, material: mat.clone(), has_window: false, has_door: true },
        WallSegment { start: [config.width, 0.0], end: [config.width, config.depth], height, material: mat.clone(), has_window: true, has_door: false },
        WallSegment { start: [config.width, config.depth], end: [0.0, config.depth], height, material: mat.clone(), has_window: true, has_door: false },
        WallSegment { start: [0.0, config.depth], end: [0.0, 0.0], height, material: mat, has_window: true, has_door: false },
    ];

    GeneratedBuilding { config: config.clone(), floors, total_rooms, total_area, exterior_walls: walls, height }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generate_basic_building() {
        let building = generate_building(&BuildingConfig::default());
        assert_eq!(building.floors.len(), 2);
        assert!(building.total_rooms > 0);
        assert!(building.height > 0.0);
    }
    #[test]
    fn room_areas() {
        assert!(RoomType::Bathroom.min_area() < RoomType::Living.min_area());
    }
}
