//! Building and construction system.
//!
//! Provides placeable structures, grid snapping, a foundation-wall-roof
//! building flow, structural integrity with support columns and weight
//! distribution, resource costs, build timers, and a blueprint system.
//!
//! # Key concepts
//!
//! - **BuildPiece**: A single building element (foundation, wall, floor, roof,
//!   stairs, door, window, pillar, etc.).
//! - **BuildGrid**: A 3D grid for snapping pieces into alignment.
//! - **Blueprint**: A saved arrangement of build pieces that can be replicated.
//! - **StructuralIntegrity**: Physics-based integrity that determines if a
//!   structure can support itself. Pieces too far from a foundation or pillar
//!   will collapse.
//! - **BuildingManager**: Top-level system managing all placed structures.

use std::collections::{HashMap, HashSet, VecDeque};

use glam::Vec3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default grid cell size for snapping (world units).
pub const DEFAULT_GRID_SIZE: f32 = 2.0;

/// Default wall height (world units).
pub const DEFAULT_WALL_HEIGHT: f32 = 3.0;

/// Maximum number of build pieces in a single structure.
pub const MAX_PIECES_PER_STRUCTURE: usize = 4096;

/// Maximum number of structures in the world.
pub const MAX_STRUCTURES: usize = 256;

/// Maximum structural integrity distance from a foundation (in grid cells).
pub const MAX_INTEGRITY_DISTANCE: usize = 8;

/// Integrity loss per grid cell from the nearest support.
pub const INTEGRITY_LOSS_PER_CELL: f32 = 0.125;

/// Minimum integrity before a piece collapses.
pub const MIN_INTEGRITY: f32 = 0.1;

/// Maximum weight a single support pillar can hold (abstract units).
pub const MAX_PILLAR_LOAD: f32 = 100.0;

/// Maximum weight a foundation can support.
pub const MAX_FOUNDATION_LOAD: f32 = 200.0;

/// Default build time in seconds (for a standard wall).
pub const DEFAULT_BUILD_TIME: f32 = 5.0;

/// Maximum blueprint size in pieces.
pub const MAX_BLUEPRINT_PIECES: usize = 512;

/// Maximum number of saved blueprints.
pub const MAX_BLUEPRINTS: usize = 64;

/// Snap tolerance distance (world units).
pub const SNAP_TOLERANCE: f32 = 0.5;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// PieceType
// ---------------------------------------------------------------------------

/// The type of a building piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceType {
    /// Ground-level base piece. Provides full structural support.
    Foundation,
    /// Vertical wall piece.
    Wall,
    /// Horizontal floor/ceiling piece.
    Floor,
    /// Angled roof piece.
    Roof,
    /// Roof ridge (peak).
    RoofRidge,
    /// Vertical support column.
    Pillar,
    /// Staircase connecting two levels.
    Stairs,
    /// A door frame (wall with opening).
    DoorFrame,
    /// A window frame (wall with window opening).
    WindowFrame,
    /// A ramp (angled floor).
    Ramp,
    /// A fence/railing piece.
    Fence,
    /// A half wall.
    HalfWall,
    /// A wedge-shaped floor (triangular).
    WedgeFloor,
    /// A wedge-shaped roof.
    WedgeRoof,
    /// A balcony platform.
    Balcony,
    /// A beam (horizontal structural support).
    Beam,
}

impl PieceType {
    /// Weight of this piece type.
    pub fn weight(&self) -> f32 {
        match self {
            Self::Foundation => 0.0, // foundations are supported by ground
            Self::Wall => 5.0,
            Self::Floor => 8.0,
            Self::Roof => 6.0,
            Self::RoofRidge => 4.0,
            Self::Pillar => 3.0,
            Self::Stairs => 7.0,
            Self::DoorFrame => 4.0,
            Self::WindowFrame => 4.0,
            Self::Ramp => 7.0,
            Self::Fence => 2.0,
            Self::HalfWall => 3.0,
            Self::WedgeFloor => 5.0,
            Self::WedgeRoof => 4.0,
            Self::Balcony => 6.0,
            Self::Beam => 4.0,
        }
    }

    /// Whether this piece type provides structural support.
    pub fn is_support(&self) -> bool {
        matches!(self, Self::Foundation | Self::Pillar | Self::Beam)
    }

    /// Whether this piece type connects to the ground.
    pub fn is_grounded(&self) -> bool {
        matches!(self, Self::Foundation)
    }

    /// Default build time for this piece.
    pub fn build_time(&self) -> f32 {
        match self {
            Self::Foundation => 8.0,
            Self::Wall => 5.0,
            Self::Floor => 6.0,
            Self::Roof => 5.0,
            Self::RoofRidge => 4.0,
            Self::Pillar => 4.0,
            Self::Stairs => 7.0,
            Self::DoorFrame => 5.0,
            Self::WindowFrame => 5.0,
            Self::Ramp => 6.0,
            Self::Fence => 3.0,
            Self::HalfWall => 3.0,
            Self::WedgeFloor => 5.0,
            Self::WedgeRoof => 4.0,
            Self::Balcony => 6.0,
            Self::Beam => 4.0,
        }
    }

    /// What piece types this can snap to.
    pub fn snap_targets(&self) -> &'static [PieceType] {
        match self {
            Self::Foundation => &[Self::Foundation],
            Self::Wall => &[Self::Foundation, Self::Floor, Self::Wall],
            Self::Floor => &[Self::Wall, Self::Pillar, Self::Beam, Self::Floor],
            Self::Roof => &[Self::Wall, Self::Roof],
            Self::RoofRidge => &[Self::Roof],
            Self::Pillar => &[Self::Foundation, Self::Floor],
            Self::Stairs => &[Self::Foundation, Self::Floor],
            Self::DoorFrame => &[Self::Foundation, Self::Floor],
            Self::WindowFrame => &[Self::Foundation, Self::Floor],
            Self::Ramp => &[Self::Foundation, Self::Floor],
            Self::Fence => &[Self::Foundation, Self::Floor, Self::Balcony],
            Self::HalfWall => &[Self::Foundation, Self::Floor],
            Self::WedgeFloor => &[Self::Floor, Self::Foundation],
            Self::WedgeRoof => &[Self::Roof, Self::Wall],
            Self::Balcony => &[Self::Wall, Self::Floor],
            Self::Beam => &[Self::Pillar, Self::Wall],
        }
    }
}

// ---------------------------------------------------------------------------
// MaterialType
// ---------------------------------------------------------------------------

/// Material used for building pieces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialType {
    Wood,
    Stone,
    Metal,
    Brick,
    Thatch,
    Concrete,
    Glass,
    Marble,
}

impl MaterialType {
    /// Durability multiplier (1.0 = standard wood baseline).
    pub fn durability_multiplier(&self) -> f32 {
        match self {
            Self::Wood => 1.0,
            Self::Stone => 2.5,
            Self::Metal => 3.0,
            Self::Brick => 2.0,
            Self::Thatch => 0.5,
            Self::Concrete => 3.5,
            Self::Glass => 0.3,
            Self::Marble => 2.8,
        }
    }

    /// Weight multiplier.
    pub fn weight_multiplier(&self) -> f32 {
        match self {
            Self::Wood => 1.0,
            Self::Stone => 2.0,
            Self::Metal => 2.5,
            Self::Brick => 1.8,
            Self::Thatch => 0.3,
            Self::Concrete => 2.2,
            Self::Glass => 0.8,
            Self::Marble => 2.3,
        }
    }

    /// Cost multiplier (relative to wood).
    pub fn cost_multiplier(&self) -> f32 {
        match self {
            Self::Wood => 1.0,
            Self::Stone => 1.5,
            Self::Metal => 3.0,
            Self::Brick => 1.3,
            Self::Thatch => 0.5,
            Self::Concrete => 2.0,
            Self::Glass => 2.5,
            Self::Marble => 4.0,
        }
    }

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wood => "Wood",
            Self::Stone => "Stone",
            Self::Metal => "Metal",
            Self::Brick => "Brick",
            Self::Thatch => "Thatch",
            Self::Concrete => "Concrete",
            Self::Glass => "Glass",
            Self::Marble => "Marble",
        }
    }
}

// ---------------------------------------------------------------------------
// ResourceCost
// ---------------------------------------------------------------------------

/// Resources required to build a piece.
#[derive(Debug, Clone)]
pub struct ResourceCost {
    /// Required materials: (item_id, quantity).
    pub materials: Vec<(String, u32)>,
    /// Optional tool requirement.
    pub required_tool: Option<String>,
    /// Skill requirement (skill_name, min_level).
    pub skill_requirement: Option<(String, u32)>,
}

impl ResourceCost {
    /// Create a new resource cost.
    pub fn new() -> Self {
        Self {
            materials: Vec::new(),
            required_tool: None,
            skill_requirement: None,
        }
    }

    /// Add a material requirement.
    pub fn material(mut self, item_id: impl Into<String>, quantity: u32) -> Self {
        self.materials.push((item_id.into(), quantity));
        self
    }

    /// Set a tool requirement.
    pub fn requires_tool(mut self, tool: impl Into<String>) -> Self {
        self.required_tool = Some(tool.into());
        self
    }

    /// Set a skill requirement.
    pub fn requires_skill(mut self, skill: impl Into<String>, level: u32) -> Self {
        self.skill_requirement = Some((skill.into(), level));
        self
    }

    /// Scale costs by a material type multiplier.
    pub fn scaled_by(&self, multiplier: f32) -> ResourceCost {
        let scaled_materials = self
            .materials
            .iter()
            .map(|(id, qty)| {
                let scaled = (*qty as f32 * multiplier).ceil() as u32;
                (id.clone(), scaled.max(1))
            })
            .collect();

        ResourceCost {
            materials: scaled_materials,
            required_tool: self.required_tool.clone(),
            skill_requirement: self.skill_requirement.clone(),
        }
    }
}

impl Default for ResourceCost {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SnapPoint
// ---------------------------------------------------------------------------

/// A connection point on a build piece where other pieces can attach.
#[derive(Debug, Clone)]
pub struct SnapPoint {
    /// Local offset from the piece origin.
    pub offset: Vec3,
    /// Direction the snap point faces.
    pub normal: Vec3,
    /// Which piece types can connect here.
    pub accepts: Vec<PieceType>,
    /// Whether this snap point is currently occupied.
    pub occupied: bool,
    /// ID of the connected piece (if any).
    pub connected_to: Option<u64>,
}

impl SnapPoint {
    /// Create a new snap point.
    pub fn new(offset: Vec3, normal: Vec3, accepts: Vec<PieceType>) -> Self {
        Self {
            offset,
            normal,
            accepts,
            occupied: false,
            connected_to: None,
        }
    }

    /// Check if a piece type can connect to this snap point.
    pub fn can_accept(&self, piece_type: PieceType) -> bool {
        !self.occupied && self.accepts.contains(&piece_type)
    }

    /// Connect a piece to this snap point.
    pub fn connect(&mut self, piece_id: u64) {
        self.occupied = true;
        self.connected_to = Some(piece_id);
    }

    /// Disconnect a piece from this snap point.
    pub fn disconnect(&mut self) {
        self.occupied = false;
        self.connected_to = None;
    }
}

// ---------------------------------------------------------------------------
// BuildPiece
// ---------------------------------------------------------------------------

/// A single placed building piece.
#[derive(Debug, Clone)]
pub struct BuildPiece {
    /// Unique piece ID.
    pub id: u64,
    /// Piece type.
    pub piece_type: PieceType,
    /// Material.
    pub material: MaterialType,
    /// World position.
    pub position: Vec3,
    /// Rotation in degrees (0, 90, 180, 270).
    pub rotation: f32,
    /// Grid coordinates (x, y, z) where y is the floor level.
    pub grid_pos: (i32, i32, i32),
    /// Structural integrity (0..1).
    pub integrity: f32,
    /// Current health / durability.
    pub health: f32,
    /// Maximum health.
    pub max_health: f32,
    /// Build progress (0..1, 1 = complete).
    pub build_progress: f32,
    /// Total build time in seconds.
    pub build_time: f32,
    /// Owner entity ID.
    pub owner: u64,
    /// Snap points on this piece.
    pub snap_points: Vec<SnapPoint>,
    /// IDs of pieces connected to this one.
    pub connected_pieces: Vec<u64>,
    /// Whether this piece is a ghost (preview, not yet placed).
    pub is_ghost: bool,
    /// The structure this piece belongs to.
    pub structure_id: u64,
    /// Effective weight (piece weight * material weight multiplier).
    pub effective_weight: f32,
    /// Whether this piece is marked for demolition.
    pub marked_for_demolition: bool,
}

impl BuildPiece {
    /// Create a new build piece.
    pub fn new(
        id: u64,
        piece_type: PieceType,
        material: MaterialType,
        position: Vec3,
        rotation: f32,
        owner: u64,
    ) -> Self {
        let base_health = 100.0 * material.durability_multiplier();
        let effective_weight = piece_type.weight() * material.weight_multiplier();

        Self {
            id,
            piece_type,
            material,
            position,
            rotation,
            grid_pos: (0, 0, 0),
            integrity: 1.0,
            health: base_health,
            max_health: base_health,
            build_progress: 0.0,
            build_time: piece_type.build_time(),
            owner,
            snap_points: Vec::new(),
            connected_pieces: Vec::new(),
            is_ghost: true,
            structure_id: 0,
            effective_weight,
            marked_for_demolition: false,
        }
    }

    /// Advance build progress. Returns true when complete.
    pub fn advance_build(&mut self, dt: f32) -> bool {
        if self.build_progress >= 1.0 {
            return true;
        }
        self.build_progress = (self.build_progress + dt / self.build_time).min(1.0);
        if self.build_progress >= 1.0 {
            self.is_ghost = false;
            return true;
        }
        false
    }

    /// Apply damage to this piece.
    pub fn damage(&mut self, amount: f32) {
        self.health = (self.health - amount).max(0.0);
    }

    /// Repair this piece.
    pub fn repair(&mut self, amount: f32) {
        self.health = (self.health + amount).min(self.max_health);
    }

    /// Whether the piece is destroyed.
    pub fn is_destroyed(&self) -> bool {
        self.health <= 0.0
    }

    /// Whether the piece is fully built.
    pub fn is_complete(&self) -> bool {
        self.build_progress >= 1.0
    }

    /// Whether the piece has sufficient structural integrity.
    pub fn is_stable(&self) -> bool {
        self.integrity >= MIN_INTEGRITY
    }

    /// Generate default snap points based on piece type.
    pub fn generate_snap_points(&mut self, grid_size: f32) {
        self.snap_points.clear();
        let half = grid_size * 0.5;

        match self.piece_type {
            PieceType::Foundation => {
                // Four edges for adjacent foundations
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(half, 0.0, 0.0),
                    Vec3::X,
                    vec![PieceType::Foundation],
                ));
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(-half, 0.0, 0.0),
                    -Vec3::X,
                    vec![PieceType::Foundation],
                ));
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(0.0, 0.0, half),
                    Vec3::Z,
                    vec![PieceType::Foundation],
                ));
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(0.0, 0.0, -half),
                    -Vec3::Z,
                    vec![PieceType::Foundation],
                ));
                // Top for walls
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(half, 0.1, 0.0),
                    Vec3::X,
                    vec![PieceType::Wall, PieceType::DoorFrame, PieceType::WindowFrame],
                ));
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(-half, 0.1, 0.0),
                    -Vec3::X,
                    vec![PieceType::Wall, PieceType::DoorFrame, PieceType::WindowFrame],
                ));
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(0.0, 0.1, half),
                    Vec3::Z,
                    vec![PieceType::Wall, PieceType::DoorFrame, PieceType::WindowFrame],
                ));
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(0.0, 0.1, -half),
                    -Vec3::Z,
                    vec![PieceType::Wall, PieceType::DoorFrame, PieceType::WindowFrame],
                ));
            }
            PieceType::Wall => {
                // Top for floor/ceiling
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(0.0, DEFAULT_WALL_HEIGHT, 0.0),
                    Vec3::Y,
                    vec![PieceType::Floor, PieceType::Roof, PieceType::Wall],
                ));
            }
            PieceType::Floor => {
                // Edges for walls
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(half, 0.0, 0.0),
                    Vec3::X,
                    vec![PieceType::Wall, PieceType::Fence, PieceType::DoorFrame],
                ));
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(-half, 0.0, 0.0),
                    -Vec3::X,
                    vec![PieceType::Wall, PieceType::Fence],
                ));
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(0.0, 0.0, half),
                    Vec3::Z,
                    vec![PieceType::Wall, PieceType::Fence],
                ));
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(0.0, 0.0, -half),
                    -Vec3::Z,
                    vec![PieceType::Wall, PieceType::Fence],
                ));
                // Corner for pillar
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(half, 0.0, half),
                    Vec3::Y,
                    vec![PieceType::Pillar],
                ));
            }
            PieceType::Pillar => {
                // Top for beams/floors
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(0.0, DEFAULT_WALL_HEIGHT, 0.0),
                    Vec3::Y,
                    vec![PieceType::Floor, PieceType::Beam],
                ));
            }
            _ => {
                // Default: one top snap point
                self.snap_points.push(SnapPoint::new(
                    Vec3::new(0.0, DEFAULT_WALL_HEIGHT, 0.0),
                    Vec3::Y,
                    vec![PieceType::Floor, PieceType::Roof],
                ));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

/// A collection of connected build pieces forming a structure.
#[derive(Debug)]
pub struct Structure {
    /// Unique structure ID.
    pub id: u64,
    /// Owner entity ID.
    pub owner: u64,
    /// Name of the structure.
    pub name: String,
    /// All pieces in this structure.
    pieces: HashMap<u64, BuildPiece>,
    /// Total weight of all pieces.
    pub total_weight: f32,
    /// Piece ID counter.
    next_piece_id: u64,
    /// Whether structural integrity needs recalculation.
    integrity_dirty: bool,
    /// Whether any piece was recently destroyed.
    has_damage: bool,
}

impl Structure {
    /// Create a new empty structure.
    pub fn new(id: u64, owner: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            owner,
            name: name.into(),
            pieces: HashMap::new(),
            total_weight: 0.0,
            next_piece_id: 1,
            integrity_dirty: true,
            has_damage: false,
        }
    }

    /// Add a piece to the structure. Returns the piece ID.
    pub fn add_piece(&mut self, mut piece: BuildPiece) -> Option<u64> {
        if self.pieces.len() >= MAX_PIECES_PER_STRUCTURE {
            return None;
        }

        let piece_id = self.next_piece_id;
        self.next_piece_id += 1;
        piece.id = piece_id;
        piece.structure_id = self.id;

        self.total_weight += piece.effective_weight;
        self.pieces.insert(piece_id, piece);
        self.integrity_dirty = true;

        Some(piece_id)
    }

    /// Remove a piece from the structure.
    pub fn remove_piece(&mut self, piece_id: u64) -> Option<BuildPiece> {
        if let Some(piece) = self.pieces.remove(&piece_id) {
            self.total_weight -= piece.effective_weight;
            // Disconnect from connected pieces
            let connected: Vec<u64> = piece.connected_pieces.clone();
            for other_id in connected {
                if let Some(other) = self.pieces.get_mut(&other_id) {
                    other.connected_pieces.retain(|&id| id != piece_id);
                    for sp in &mut other.snap_points {
                        if sp.connected_to == Some(piece_id) {
                            sp.disconnect();
                        }
                    }
                }
            }
            self.integrity_dirty = true;
            Some(piece)
        } else {
            None
        }
    }

    /// Get a piece by ID.
    pub fn get_piece(&self, piece_id: u64) -> Option<&BuildPiece> {
        self.pieces.get(&piece_id)
    }

    /// Get a mutable piece by ID.
    pub fn get_piece_mut(&mut self, piece_id: u64) -> Option<&mut BuildPiece> {
        self.pieces.get_mut(&piece_id)
    }

    /// Get all pieces.
    pub fn pieces(&self) -> &HashMap<u64, BuildPiece> {
        &self.pieces
    }

    /// Piece count.
    pub fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    /// Connect two pieces.
    pub fn connect_pieces(&mut self, piece_a: u64, piece_b: u64) -> bool {
        if !self.pieces.contains_key(&piece_a) || !self.pieces.contains_key(&piece_b) {
            return false;
        }

        if let Some(a) = self.pieces.get_mut(&piece_a) {
            if !a.connected_pieces.contains(&piece_b) {
                a.connected_pieces.push(piece_b);
            }
        }
        if let Some(b) = self.pieces.get_mut(&piece_b) {
            if !b.connected_pieces.contains(&piece_a) {
                b.connected_pieces.push(piece_a);
            }
        }

        self.integrity_dirty = true;
        true
    }

    // -----------------------------------------------------------------------
    // Structural integrity
    // -----------------------------------------------------------------------

    /// Recalculate structural integrity for all pieces.
    pub fn recalculate_integrity(&mut self) {
        if !self.integrity_dirty {
            return;
        }

        // Find all support pieces (foundations, pillars)
        let support_ids: Vec<u64> = self
            .pieces
            .iter()
            .filter(|(_, p)| p.piece_type.is_support())
            .map(|(id, _)| *id)
            .collect();

        // BFS from supports, assigning integrity based on distance
        let mut visited: HashMap<u64, usize> = HashMap::new();
        let mut queue: VecDeque<(u64, usize)> = VecDeque::new();

        for &support_id in &support_ids {
            if let Some(piece) = self.pieces.get(&support_id) {
                if piece.piece_type.is_grounded() {
                    queue.push_back((support_id, 0));
                    visited.insert(support_id, 0);
                } else {
                    // Pillars need to be connected to something grounded
                    queue.push_back((support_id, 1));
                    visited.insert(support_id, 1);
                }
            }
        }

        while let Some((current_id, distance)) = queue.pop_front() {
            if distance > MAX_INTEGRITY_DISTANCE {
                continue;
            }

            let connected = match self.pieces.get(&current_id) {
                Some(piece) => piece.connected_pieces.clone(),
                None => continue,
            };

            for neighbor_id in connected {
                let new_distance = distance + 1;
                let update = match visited.get(&neighbor_id) {
                    Some(&existing) => new_distance < existing,
                    None => true,
                };

                if update {
                    visited.insert(neighbor_id, new_distance);
                    queue.push_back((neighbor_id, new_distance));
                }
            }
        }

        // Assign integrity values
        for (id, piece) in &mut self.pieces {
            if piece.piece_type.is_grounded() {
                piece.integrity = 1.0;
            } else if let Some(&distance) = visited.get(id) {
                piece.integrity =
                    (1.0 - distance as f32 * INTEGRITY_LOSS_PER_CELL).max(0.0);
            } else {
                // Not connected to any support — will collapse
                piece.integrity = 0.0;
            }
        }

        self.integrity_dirty = false;
    }

    /// Find pieces that should collapse (integrity too low).
    pub fn find_collapsing_pieces(&self) -> Vec<u64> {
        self.pieces
            .iter()
            .filter(|(_, p)| !p.piece_type.is_grounded() && p.integrity < MIN_INTEGRITY && p.is_complete())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Process collapses: remove pieces with no integrity.
    pub fn process_collapses(&mut self) -> Vec<BuildPiece> {
        self.recalculate_integrity();
        let collapsing = self.find_collapsing_pieces();
        let mut collapsed = Vec::new();

        for id in collapsing {
            if let Some(piece) = self.remove_piece(id) {
                collapsed.push(piece);
            }
        }

        if !collapsed.is_empty() {
            // Recalculate after removals (chain collapses)
            self.integrity_dirty = true;
            self.recalculate_integrity();
        }

        collapsed
    }

    /// Update build progress on all pieces.
    pub fn update_building(&mut self, dt: f32) {
        for piece in self.pieces.values_mut() {
            if !piece.is_complete() {
                piece.advance_build(dt);
            }
        }
    }

    /// Apply damage to a specific piece.
    pub fn damage_piece(&mut self, piece_id: u64, amount: f32) {
        if let Some(piece) = self.pieces.get_mut(&piece_id) {
            piece.damage(amount);
            if piece.is_destroyed() {
                self.has_damage = true;
            }
        }
    }

    /// Remove destroyed pieces and process collapses.
    pub fn cleanup_destroyed(&mut self) -> Vec<BuildPiece> {
        let destroyed_ids: Vec<u64> = self
            .pieces
            .iter()
            .filter(|(_, p)| p.is_destroyed())
            .map(|(id, _)| *id)
            .collect();

        let mut removed = Vec::new();
        for id in destroyed_ids {
            if let Some(piece) = self.remove_piece(id) {
                removed.push(piece);
            }
        }

        // Chain collapses
        if !removed.is_empty() {
            let mut collapsed = self.process_collapses();
            removed.append(&mut collapsed);
        }

        self.has_damage = false;
        removed
    }

    /// Get the bounding box of the structure (min, max).
    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);

        for piece in self.pieces.values() {
            min = min.min(piece.position);
            max = max.max(piece.position + Vec3::new(DEFAULT_GRID_SIZE, DEFAULT_WALL_HEIGHT, DEFAULT_GRID_SIZE));
        }

        (min, max)
    }
}

// ---------------------------------------------------------------------------
// BlueprintPiece
// ---------------------------------------------------------------------------

/// A piece definition within a blueprint.
#[derive(Debug, Clone)]
pub struct BlueprintPiece {
    /// Piece type.
    pub piece_type: PieceType,
    /// Material.
    pub material: MaterialType,
    /// Offset from the blueprint origin.
    pub offset: Vec3,
    /// Rotation.
    pub rotation: f32,
    /// Grid position relative to blueprint origin.
    pub grid_offset: (i32, i32, i32),
}

// ---------------------------------------------------------------------------
// Blueprint
// ---------------------------------------------------------------------------

/// A saved building design that can be replicated.
#[derive(Debug, Clone)]
pub struct Blueprint {
    /// Unique blueprint ID.
    pub id: u32,
    /// Display name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Creator entity ID.
    pub creator: u64,
    /// Piece definitions.
    pub pieces: Vec<BlueprintPiece>,
    /// Total resource cost.
    pub total_cost: Vec<(String, u32)>,
    /// Category tag.
    pub category: String,
    /// Creation timestamp.
    pub created_at: f64,
}

impl Blueprint {
    /// Create a new empty blueprint.
    pub fn new(id: u32, name: impl Into<String>, creator: u64) -> Self {
        Self {
            id,
            name: name.into(),
            description: String::new(),
            creator,
            pieces: Vec::new(),
            total_cost: Vec::new(),
            category: "General".into(),
            created_at: 0.0,
        }
    }

    /// Add a piece to the blueprint.
    pub fn add_piece(&mut self, piece: BlueprintPiece) -> bool {
        if self.pieces.len() >= MAX_BLUEPRINT_PIECES {
            return false;
        }
        self.pieces.push(piece);
        true
    }

    /// Create a blueprint from an existing structure.
    pub fn from_structure(
        id: u32,
        name: impl Into<String>,
        structure: &Structure,
    ) -> Self {
        let mut blueprint = Self::new(id, name, structure.owner);

        // Find the minimum position as origin
        let origin = structure
            .pieces()
            .values()
            .map(|p| p.position)
            .fold(Vec3::splat(f32::MAX), |a, b| a.min(b));

        for piece in structure.pieces().values() {
            let bp_piece = BlueprintPiece {
                piece_type: piece.piece_type,
                material: piece.material,
                offset: piece.position - origin,
                rotation: piece.rotation,
                grid_offset: (
                    piece.grid_pos.0,
                    piece.grid_pos.1,
                    piece.grid_pos.2,
                ),
            };
            blueprint.add_piece(bp_piece);
        }

        blueprint
    }

    /// Get the piece count.
    pub fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    /// Calculate total resource costs.
    pub fn calculate_costs(&mut self, base_costs: &HashMap<PieceType, ResourceCost>) {
        let mut combined: HashMap<String, u32> = HashMap::new();

        for piece in &self.pieces {
            if let Some(cost) = base_costs.get(&piece.piece_type) {
                let scaled = cost.scaled_by(piece.material.cost_multiplier());
                for (item_id, qty) in &scaled.materials {
                    *combined.entry(item_id.clone()).or_insert(0) += qty;
                }
            }
        }

        self.total_cost = combined.into_iter().collect();
    }
}

// ---------------------------------------------------------------------------
// PlacementValidation
// ---------------------------------------------------------------------------

/// Result of validating a piece placement.
#[derive(Debug, Clone)]
pub enum PlacementResult {
    /// Placement is valid.
    Valid,
    /// No valid snap point found.
    NoSnapPoint,
    /// Would overlap an existing piece.
    Overlapping,
    /// Missing required foundation below.
    NoFoundation,
    /// Insufficient resources.
    InsufficientResources(Vec<(String, u32)>),
    /// Terrain is not suitable.
    BadTerrain,
    /// Exceeds maximum pieces per structure.
    StructureFull,
    /// Not enough structural support.
    InsufficientSupport,
    /// Blocked by another entity.
    Blocked,
}

// ---------------------------------------------------------------------------
// BuildingManager
// ---------------------------------------------------------------------------

/// Top-level manager for all building operations.
pub struct BuildingManager {
    /// All structures.
    structures: HashMap<u64, Structure>,
    /// Saved blueprints.
    blueprints: Vec<Blueprint>,
    /// Next structure ID.
    next_structure_id: u64,
    /// Next blueprint ID.
    next_blueprint_id: u32,
    /// Grid size used for snapping.
    pub grid_size: f32,
    /// Base costs for each piece type.
    base_costs: HashMap<PieceType, ResourceCost>,
    /// Building events.
    events: Vec<BuildingEvent>,
}

/// Events emitted by the building system.
#[derive(Debug, Clone)]
pub enum BuildingEvent {
    /// A piece was placed.
    PiecePlaced {
        structure_id: u64,
        piece_id: u64,
        piece_type: PieceType,
        position: Vec3,
    },
    /// A piece was removed.
    PieceRemoved {
        structure_id: u64,
        piece_id: u64,
    },
    /// A piece collapsed due to lack of support.
    PieceCollapsed {
        structure_id: u64,
        piece_id: u64,
        piece_type: PieceType,
    },
    /// A piece was completed.
    PieceCompleted {
        structure_id: u64,
        piece_id: u64,
    },
    /// A piece was destroyed.
    PieceDestroyed {
        structure_id: u64,
        piece_id: u64,
    },
    /// A structure was created.
    StructureCreated {
        structure_id: u64,
        owner: u64,
    },
    /// A structure was demolished.
    StructureDemolished {
        structure_id: u64,
    },
}

impl BuildingManager {
    /// Create a new building manager.
    pub fn new() -> Self {
        Self {
            structures: HashMap::new(),
            blueprints: Vec::new(),
            next_structure_id: 1,
            next_blueprint_id: 1,
            grid_size: DEFAULT_GRID_SIZE,
            base_costs: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Set the base cost for a piece type.
    pub fn set_base_cost(&mut self, piece_type: PieceType, cost: ResourceCost) {
        self.base_costs.insert(piece_type, cost);
    }

    /// Create a new structure.
    pub fn create_structure(
        &mut self,
        owner: u64,
        name: impl Into<String>,
    ) -> u64 {
        let id = self.next_structure_id;
        self.next_structure_id += 1;

        let structure = Structure::new(id, owner, name);
        self.structures.insert(id, structure);

        self.events.push(BuildingEvent::StructureCreated {
            structure_id: id,
            owner,
        });

        id
    }

    /// Get a structure by ID.
    pub fn get_structure(&self, id: u64) -> Option<&Structure> {
        self.structures.get(&id)
    }

    /// Get a mutable structure by ID.
    pub fn get_structure_mut(&mut self, id: u64) -> Option<&mut Structure> {
        self.structures.get_mut(&id)
    }

    /// Demolish a structure.
    pub fn demolish_structure(&mut self, id: u64) -> bool {
        if self.structures.remove(&id).is_some() {
            self.events
                .push(BuildingEvent::StructureDemolished { structure_id: id });
            true
        } else {
            false
        }
    }

    /// Snap a position to the build grid.
    pub fn snap_to_grid(&self, position: Vec3) -> Vec3 {
        Vec3::new(
            (position.x / self.grid_size).round() * self.grid_size,
            (position.y / self.grid_size).round() * self.grid_size,
            (position.z / self.grid_size).round() * self.grid_size,
        )
    }

    /// Place a piece in a structure.
    pub fn place_piece(
        &mut self,
        structure_id: u64,
        piece_type: PieceType,
        material: MaterialType,
        position: Vec3,
        rotation: f32,
        owner: u64,
    ) -> Result<u64, PlacementResult> {
        let structure = self
            .structures
            .get_mut(&structure_id)
            .ok_or(PlacementResult::BadTerrain)?;

        if structure.piece_count() >= MAX_PIECES_PER_STRUCTURE {
            return Err(PlacementResult::StructureFull);
        }

        let snapped = self.snap_to_grid(position);
        let grid_pos = (
            (snapped.x / self.grid_size) as i32,
            (snapped.y / self.grid_size) as i32,
            (snapped.z / self.grid_size) as i32,
        );

        // Check for overlapping pieces
        let overlapping = structure.pieces().values().any(|p| {
            p.grid_pos == grid_pos && p.piece_type == piece_type
        });
        if overlapping {
            return Err(PlacementResult::Overlapping);
        }

        // Foundation check for non-foundation pieces
        if !piece_type.is_grounded() && grid_pos.1 == 0 {
            let has_foundation = structure.pieces().values().any(|p| {
                p.piece_type.is_grounded()
                    && p.grid_pos.0 == grid_pos.0
                    && p.grid_pos.2 == grid_pos.2
            });
            if !has_foundation && !piece_type.is_support() {
                return Err(PlacementResult::NoFoundation);
            }
        }

        let mut piece = BuildPiece::new(0, piece_type, material, snapped, rotation, owner);
        piece.grid_pos = grid_pos;
        piece.generate_snap_points(self.grid_size);

        let piece_id = structure
            .add_piece(piece)
            .ok_or(PlacementResult::StructureFull)?;

        // Auto-connect to adjacent pieces
        let connected: Vec<u64> = structure
            .pieces()
            .iter()
            .filter(|(&id, p)| {
                id != piece_id
                    && (p.grid_pos.0 - grid_pos.0).abs() <= 1
                    && (p.grid_pos.1 - grid_pos.1).abs() <= 1
                    && (p.grid_pos.2 - grid_pos.2).abs() <= 1
            })
            .map(|(&id, _)| id)
            .collect();

        for other_id in connected {
            structure.connect_pieces(piece_id, other_id);
        }

        self.events.push(BuildingEvent::PiecePlaced {
            structure_id,
            piece_id,
            piece_type,
            position: snapped,
        });

        Ok(piece_id)
    }

    /// Save a blueprint from a structure.
    pub fn save_blueprint(
        &mut self,
        structure_id: u64,
        name: impl Into<String>,
    ) -> Option<u32> {
        let structure = self.structures.get(&structure_id)?;

        if self.blueprints.len() >= MAX_BLUEPRINTS {
            return None;
        }

        let bp_id = self.next_blueprint_id;
        self.next_blueprint_id += 1;

        let mut blueprint = Blueprint::from_structure(bp_id, name, structure);
        blueprint.calculate_costs(&self.base_costs);
        self.blueprints.push(blueprint);

        Some(bp_id)
    }

    /// Get a blueprint by ID.
    pub fn get_blueprint(&self, id: u32) -> Option<&Blueprint> {
        self.blueprints.iter().find(|b| b.id == id)
    }

    /// Get all blueprints.
    pub fn blueprints(&self) -> &[Blueprint] {
        &self.blueprints
    }

    /// Update all structures (build progress, collapses).
    pub fn update(&mut self, dt: f32) {
        let mut all_collapses = Vec::new();
        let mut all_completed = Vec::new();

        for (structure_id, structure) in &mut self.structures {
            // Update building
            let prev_incomplete: HashSet<u64> = structure
                .pieces()
                .iter()
                .filter(|(_, p)| !p.is_complete())
                .map(|(id, _)| *id)
                .collect();

            structure.update_building(dt);

            // Detect newly completed pieces
            for id in &prev_incomplete {
                if let Some(piece) = structure.get_piece(*id) {
                    if piece.is_complete() {
                        all_completed.push((*structure_id, *id));
                    }
                }
            }

            // Process collapses
            let collapsed = structure.cleanup_destroyed();
            for piece in &collapsed {
                all_collapses.push((*structure_id, piece.id, piece.piece_type));
            }
        }

        for (structure_id, piece_id) in all_completed {
            self.events.push(BuildingEvent::PieceCompleted {
                structure_id,
                piece_id,
            });
        }

        for (structure_id, piece_id, piece_type) in all_collapses {
            self.events.push(BuildingEvent::PieceCollapsed {
                structure_id,
                piece_id,
                piece_type,
            });
        }
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<BuildingEvent> {
        std::mem::take(&mut self.events)
    }

    /// Get structure count.
    pub fn structure_count(&self) -> usize {
        self.structures.len()
    }

    /// Find the structure at a world position.
    pub fn structure_at(&self, position: Vec3) -> Option<u64> {
        for (id, structure) in &self.structures {
            let (min, max) = structure.bounding_box();
            if position.x >= min.x
                && position.x <= max.x
                && position.y >= min.y
                && position.y <= max.y
                && position.z >= min.z
                && position.z <= max.z
            {
                return Some(*id);
            }
        }
        None
    }
}

impl Default for BuildingManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_place_foundation_and_wall() {
        let mut mgr = BuildingManager::new();
        let sid = mgr.create_structure(1, "Test House");

        // Place foundation
        let fid = mgr.place_piece(
            sid,
            PieceType::Foundation,
            MaterialType::Stone,
            Vec3::new(0.0, 0.0, 0.0),
            0.0,
            1,
        );
        assert!(fid.is_ok());

        // Place wall on foundation
        let wid = mgr.place_piece(
            sid,
            PieceType::Wall,
            MaterialType::Wood,
            Vec3::new(1.0, 0.0, 0.0),
            0.0,
            1,
        );
        assert!(wid.is_ok());
    }

    #[test]
    fn test_structural_integrity() {
        let mut structure = Structure::new(1, 1, "Test");

        // Add a foundation
        let mut foundation = BuildPiece::new(
            0, PieceType::Foundation, MaterialType::Stone,
            Vec3::ZERO, 0.0, 1,
        );
        foundation.grid_pos = (0, 0, 0);
        foundation.build_progress = 1.0;
        foundation.is_ghost = false;
        let fid = structure.add_piece(foundation).unwrap();

        // Add connected floor
        let mut floor = BuildPiece::new(
            0, PieceType::Floor, MaterialType::Wood,
            Vec3::new(2.0, 3.0, 0.0), 0.0, 1,
        );
        floor.grid_pos = (1, 1, 0);
        floor.build_progress = 1.0;
        floor.is_ghost = false;
        let floor_id = structure.add_piece(floor).unwrap();

        structure.connect_pieces(fid, floor_id);
        structure.recalculate_integrity();

        let f_piece = structure.get_piece(fid).unwrap();
        assert_eq!(f_piece.integrity, 1.0);

        let fl_piece = structure.get_piece(floor_id).unwrap();
        assert!(fl_piece.integrity > 0.0);
        assert!(fl_piece.integrity < 1.0);
    }

    #[test]
    fn test_blueprint() {
        let mut mgr = BuildingManager::new();
        let sid = mgr.create_structure(1, "Template");

        mgr.place_piece(sid, PieceType::Foundation, MaterialType::Stone,
            Vec3::ZERO, 0.0, 1).unwrap();
        mgr.place_piece(sid, PieceType::Wall, MaterialType::Wood,
            Vec3::new(2.0, 0.0, 0.0), 0.0, 1).unwrap();

        let bp_id = mgr.save_blueprint(sid, "Small House").unwrap();
        let bp = mgr.get_blueprint(bp_id).unwrap();
        assert_eq!(bp.piece_count(), 2);
    }

    #[test]
    fn test_snap_to_grid() {
        let mgr = BuildingManager::new();
        let snapped = mgr.snap_to_grid(Vec3::new(3.3, 1.7, 5.8));
        assert!((snapped.x - 4.0).abs() < EPSILON);
        assert!((snapped.y - 2.0).abs() < EPSILON);
        assert!((snapped.z - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_material_properties() {
        assert!(MaterialType::Metal.durability_multiplier() > MaterialType::Wood.durability_multiplier());
        assert!(MaterialType::Thatch.weight_multiplier() < MaterialType::Stone.weight_multiplier());
    }
}
