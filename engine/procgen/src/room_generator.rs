// engine/procgen/src/room_generator.rs
//
// Room and interior generation for the Genovo engine.
//
// Generates room shapes, furniture placement, door/window positions,
// hallway connections, and decoration.

use std::collections::HashMap;

pub const MIN_ROOM_SIZE: f32 = 3.0;
pub const MAX_ROOM_SIZE: f32 = 15.0;
pub const DEFAULT_WALL_HEIGHT: f32 = 3.0;
pub const DEFAULT_DOOR_WIDTH: f32 = 1.0;
pub const DEFAULT_WINDOW_WIDTH: f32 = 1.2;
pub const HALLWAY_WIDTH: f32 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoomShape { Rectangle, LShape, TShape, UShape, Circular, Irregular }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoomType { Living, Bedroom, Kitchen, Bathroom, Office, Storage, Hallway, Entrance, Stairwell, Balcony }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WallSide { North, South, East, West }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FurnitureType { Bed, Table, Chair, Sofa, Desk, Shelf, Wardrobe, Toilet, Sink, Stove, Fridge, Plant, Lamp, Rug, Painting }

impl FurnitureType {
    pub fn size(&self) -> [f32; 2] {
        match self { Self::Bed => [2.0, 1.4], Self::Table => [1.2, 0.8], Self::Chair => [0.5, 0.5], Self::Sofa => [2.0, 0.8],
            Self::Desk => [1.4, 0.7], Self::Shelf => [1.0, 0.3], Self::Wardrobe => [1.2, 0.6], Self::Toilet => [0.6, 0.4],
            Self::Sink => [0.6, 0.4], Self::Stove => [0.6, 0.6], Self::Fridge => [0.7, 0.7], Self::Plant => [0.4, 0.4],
            Self::Lamp => [0.3, 0.3], Self::Rug => [2.0, 1.5], Self::Painting => [0.8, 0.1] }
    }
    pub fn against_wall(&self) -> bool { matches!(self, Self::Bed|Self::Sofa|Self::Shelf|Self::Wardrobe|Self::Desk|Self::Toilet|Self::Sink|Self::Stove|Self::Fridge|Self::Painting) }
}

#[derive(Debug, Clone)]
pub struct FurniturePlacement { pub furniture_type: FurnitureType, pub position: [f32; 2], pub rotation: f32, pub wall: Option<WallSide> }

#[derive(Debug, Clone)]
pub struct DoorPlacement { pub wall: WallSide, pub offset: f32, pub width: f32, pub connects_to: Option<usize> }

#[derive(Debug, Clone)]
pub struct WindowPlacement { pub wall: WallSide, pub offset: f32, pub width: f32, pub height: f32, pub sill_height: f32 }

#[derive(Debug, Clone)]
pub struct Room {
    pub id: usize,
    pub room_type: RoomType,
    pub shape: RoomShape,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub wall_height: f32,
    pub doors: Vec<DoorPlacement>,
    pub windows: Vec<WindowPlacement>,
    pub furniture: Vec<FurniturePlacement>,
    pub floor_material: String,
    pub wall_material: String,
    pub ceiling_material: String,
}

impl Room {
    pub fn new(id: usize, room_type: RoomType, position: [f32; 2], size: [f32; 2]) -> Self {
        Self { id, room_type, shape: RoomShape::Rectangle, position, size, wall_height: DEFAULT_WALL_HEIGHT,
            doors: Vec::new(), windows: Vec::new(), furniture: Vec::new(),
            floor_material: "wood".to_string(), wall_material: "plaster".to_string(), ceiling_material: "plaster".to_string() }
    }

    pub fn area(&self) -> f32 { self.size[0] * self.size[1] }
    pub fn center(&self) -> [f32; 2] { [self.position[0] + self.size[0] * 0.5, self.position[1] + self.size[1] * 0.5] }
    pub fn wall_length(&self, side: WallSide) -> f32 { match side { WallSide::North | WallSide::South => self.size[0], WallSide::East | WallSide::West => self.size[1] } }

    pub fn add_door(&mut self, wall: WallSide, offset: f32, connects_to: Option<usize>) {
        self.doors.push(DoorPlacement { wall, offset, width: DEFAULT_DOOR_WIDTH, connects_to });
    }

    pub fn add_window(&mut self, wall: WallSide, offset: f32) {
        self.windows.push(WindowPlacement { wall, offset, width: DEFAULT_WINDOW_WIDTH, height: 1.2, sill_height: 0.9 });
    }
}

#[derive(Debug, Clone)]
pub struct Hallway { pub start: [f32; 2], pub end: [f32; 2], pub width: f32, pub room_a: usize, pub room_b: usize }

impl Hallway {
    pub fn length(&self) -> f32 { let dx = self.end[0] - self.start[0]; let dz = self.end[1] - self.start[1]; (dx*dx + dz*dz).sqrt() }
}

#[derive(Debug, Clone)]
pub struct GeneratedInterior { pub rooms: Vec<Room>, pub hallways: Vec<Hallway>, pub bounds_min: [f32; 2], pub bounds_max: [f32; 2] }

impl GeneratedInterior {
    pub fn room_count(&self) -> usize { self.rooms.len() }
    pub fn total_area(&self) -> f32 { self.rooms.iter().map(|r| r.area()).sum() }
}

#[derive(Debug, Clone)]
pub struct RoomGeneratorConfig {
    pub room_count: usize,
    pub min_room_size: [f32; 2],
    pub max_room_size: [f32; 2],
    pub wall_height: f32,
    pub hallway_width: f32,
    pub exterior_windows: bool,
    pub furnish: bool,
    pub seed: u64,
    pub room_types: Vec<RoomType>,
}

impl Default for RoomGeneratorConfig {
    fn default() -> Self {
        Self { room_count: 6, min_room_size: [3.0, 3.0], max_room_size: [8.0, 8.0], wall_height: DEFAULT_WALL_HEIGHT,
            hallway_width: HALLWAY_WIDTH, exterior_windows: true, furnish: true, seed: 42,
            room_types: vec![RoomType::Living, RoomType::Bedroom, RoomType::Kitchen, RoomType::Bathroom, RoomType::Office, RoomType::Storage] }
    }
}

pub struct RoomGenerator { pub config: RoomGeneratorConfig, rng_state: u64 }

impl RoomGenerator {
    pub fn new(config: RoomGeneratorConfig) -> Self { Self { rng_state: config.seed, config } }

    fn rand(&mut self) -> f32 { self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.rng_state >> 33) as f32) / (u32::MAX as f32) }
    fn rand_range(&mut self, min: f32, max: f32) -> f32 { min + self.rand() * (max - min) }

    pub fn generate(&mut self) -> GeneratedInterior {
        let mut rooms = Vec::new();
        let mut pos = [0.0f32, 0.0f32];

        for i in 0..self.config.room_count {
            let w = self.rand_range(self.config.min_room_size[0], self.config.max_room_size[0]);
            let h = self.rand_range(self.config.min_room_size[1], self.config.max_room_size[1]);
            let rt = self.config.room_types[i % self.config.room_types.len()];
            let mut room = Room::new(i, rt, pos, [w, h]);
            room.wall_height = self.config.wall_height;
            self.set_materials(&mut room);
            rooms.push(room);

            // Place next room adjacent.
            if self.rand() > 0.5 { pos[0] += w + self.config.hallway_width; }
            else { pos[1] += h + self.config.hallway_width; }
        }

        // Connect rooms with doors/hallways.
        let mut hallways = Vec::new();
        for i in 0..rooms.len().saturating_sub(1) {
            let ca = rooms[i].center();
            let cb = rooms[i + 1].center();
            rooms[i].add_door(WallSide::East, rooms[i].size[1] * 0.5, Some(i + 1));
            rooms[i + 1].add_door(WallSide::West, rooms[i + 1].size[1] * 0.5, Some(i));
            hallways.push(Hallway { start: ca, end: cb, width: self.config.hallway_width, room_a: i, room_b: i + 1 });
        }

        // Add windows to exterior walls.
        if self.config.exterior_windows {
            for room in &mut rooms {
                let walls = [WallSide::North, WallSide::South, WallSide::East, WallSide::West];
                for &wall in &walls {
                    let wl = room.wall_length(wall);
                    if wl > 3.0 && !room.doors.iter().any(|d| d.wall == wall) {
                        room.add_window(wall, wl * 0.5);
                    }
                }
            }
        }

        // Furnish rooms.
        if self.config.furnish {
            for room in &mut rooms {
                self.furnish_room(room);
            }
        }

        let mut bmin = [f32::MAX; 2]; let mut bmax = [f32::MIN; 2];
        for r in &rooms {
            bmin[0] = bmin[0].min(r.position[0]); bmin[1] = bmin[1].min(r.position[1]);
            bmax[0] = bmax[0].max(r.position[0] + r.size[0]); bmax[1] = bmax[1].max(r.position[1] + r.size[1]);
        }

        GeneratedInterior { rooms, hallways, bounds_min: bmin, bounds_max: bmax }
    }

    fn set_materials(&self, room: &mut Room) {
        match room.room_type {
            RoomType::Kitchen => { room.floor_material = "tile".to_string(); }
            RoomType::Bathroom => { room.floor_material = "tile".to_string(); room.wall_material = "tile".to_string(); }
            RoomType::Bedroom => { room.floor_material = "carpet".to_string(); }
            _ => {}
        }
    }

    fn furnish_room(&mut self, room: &mut Room) {
        let furniture_list = match room.room_type {
            RoomType::Bedroom => vec![FurnitureType::Bed, FurnitureType::Wardrobe, FurnitureType::Lamp],
            RoomType::Living => vec![FurnitureType::Sofa, FurnitureType::Table, FurnitureType::Lamp, FurnitureType::Plant],
            RoomType::Kitchen => vec![FurnitureType::Stove, FurnitureType::Fridge, FurnitureType::Table, FurnitureType::Chair],
            RoomType::Bathroom => vec![FurnitureType::Toilet, FurnitureType::Sink],
            RoomType::Office => vec![FurnitureType::Desk, FurnitureType::Chair, FurnitureType::Shelf],
            RoomType::Storage => vec![FurnitureType::Shelf, FurnitureType::Shelf],
            _ => vec![],
        };

        let mut used_positions: Vec<([f32; 2], [f32; 2])> = Vec::new();
        for ft in furniture_list {
            let size = ft.size();
            let mut attempts = 0;
            loop {
                let x = self.rand_range(0.3, room.size[0] - size[0] - 0.3);
                let z = self.rand_range(0.3, room.size[1] - size[1] - 0.3);
                let pos = [room.position[0] + x, room.position[1] + z];
                let overlaps = used_positions.iter().any(|(p, s)| pos[0] < p[0]+s[0] && pos[0]+size[0] > p[0] && pos[1] < p[1]+s[1] && pos[1]+size[1] > p[1]);
                if !overlaps || attempts > 20 {
                    if !overlaps {
                        let wall = if ft.against_wall() { Some(WallSide::North) } else { None };
                        room.furniture.push(FurniturePlacement { furniture_type: ft, position: pos, rotation: 0.0, wall });
                        used_positions.push((pos, size));
                    }
                    break;
                }
                attempts += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_room_creation() { let r = Room::new(0, RoomType::Living, [0.0, 0.0], [5.0, 4.0]); assert!((r.area() - 20.0).abs() < 0.01); }
    #[test] fn test_generator() { let mut gen = RoomGenerator::new(RoomGeneratorConfig::default()); let interior = gen.generate(); assert_eq!(interior.room_count(), 6); assert!(interior.total_area() > 0.0); }
    #[test] fn test_furniture_size() { let s = FurnitureType::Bed.size(); assert!(s[0] > 1.0 && s[1] > 1.0); }
}
