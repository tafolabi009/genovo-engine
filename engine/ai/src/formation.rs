//! Group AI formation system.
//!
//! Manages formations where a group of units moves together in a defined
//! geometric pattern relative to a leader. Supports standard military
//! formations (line, column, wedge, circle, grid) as well as arbitrary
//! custom shapes.
//!
//! # Key concepts
//!
//! - **FormationShape**: the geometric pattern (line, column, wedge, etc.).
//! - **FormationSlot**: a position within the formation, defined as an
//!   offset from the leader.
//! - **Formation**: a shape plus the assignment of units to slots.
//! - **FormationManager**: creates, updates, and manages active formations.
//!
//! # Slot assignment
//!
//! When units join or leave a formation, slots are reassigned using a
//! greedy nearest-neighbor algorithm: each unassigned unit is matched to
//! the closest unoccupied slot, minimizing total travel distance.

use glam::{Mat2, Vec2};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default spacing between units (world units).
pub const DEFAULT_SPACING: f32 = 2.0;

/// Default wedge half-angle (degrees).
pub const DEFAULT_WEDGE_ANGLE: f32 = 45.0;

/// Maximum units in a single formation.
pub const MAX_FORMATION_UNITS: usize = 128;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// FormationShape
// ---------------------------------------------------------------------------

/// The geometric shape of a formation.
#[derive(Debug, Clone)]
pub enum FormationShape {
    /// Units arranged in a horizontal row perpendicular to the leader's
    /// forward direction.
    Line {
        /// Spacing between adjacent units.
        spacing: f32,
    },
    /// Units arranged in a column behind the leader.
    Column {
        /// Spacing between units along the column.
        spacing: f32,
    },
    /// V-shaped formation (wedge/chevron).
    Wedge {
        /// Half-angle of the wedge in degrees.
        angle: f32,
        /// Spacing between units along each arm.
        spacing: f32,
    },
    /// Units arranged in a circle around the leader.
    Circle {
        /// Radius of the circle.
        radius: f32,
    },
    /// Rectangular grid behind the leader.
    Grid {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
        /// Spacing between units.
        spacing: f32,
    },
    /// Arbitrary positions defined by explicit offsets.
    Custom {
        /// Per-slot offsets relative to the leader.
        offsets: Vec<Vec2>,
    },
}

impl FormationShape {
    /// Create a line formation with default spacing.
    pub fn line() -> Self {
        FormationShape::Line {
            spacing: DEFAULT_SPACING,
        }
    }

    /// Create a column formation with default spacing.
    pub fn column() -> Self {
        FormationShape::Column {
            spacing: DEFAULT_SPACING,
        }
    }

    /// Create a wedge formation with default angle and spacing.
    pub fn wedge() -> Self {
        FormationShape::Wedge {
            angle: DEFAULT_WEDGE_ANGLE,
            spacing: DEFAULT_SPACING,
        }
    }

    /// Create a circle formation.
    pub fn circle(radius: f32) -> Self {
        FormationShape::Circle { radius }
    }

    /// Create a grid formation.
    pub fn grid(rows: usize, cols: usize) -> Self {
        FormationShape::Grid {
            rows,
            cols,
            spacing: DEFAULT_SPACING,
        }
    }

    /// Generate slot offsets for `unit_count` units.
    ///
    /// Offsets are relative to the leader's position in local space (X = right,
    /// Y = forward). The leader itself is not included.
    pub fn generate_offsets(&self, unit_count: usize) -> Vec<Vec2> {
        match self {
            FormationShape::Line { spacing } => {
                Self::generate_line_offsets(unit_count, *spacing)
            }
            FormationShape::Column { spacing } => {
                Self::generate_column_offsets(unit_count, *spacing)
            }
            FormationShape::Wedge { angle, spacing } => {
                Self::generate_wedge_offsets(unit_count, *angle, *spacing)
            }
            FormationShape::Circle { radius } => {
                Self::generate_circle_offsets(unit_count, *radius)
            }
            FormationShape::Grid { rows, cols, spacing } => {
                Self::generate_grid_offsets(unit_count, *rows, *cols, *spacing)
            }
            FormationShape::Custom { offsets } => {
                // Return as many offsets as we have, or pad with zeros.
                let mut result = offsets.clone();
                while result.len() < unit_count {
                    result.push(Vec2::ZERO);
                }
                result.truncate(unit_count);
                result
            }
        }
    }

    /// Generate offsets for a line formation (spread left-right).
    fn generate_line_offsets(count: usize, spacing: f32) -> Vec<Vec2> {
        let mut offsets = Vec::with_capacity(count);
        for i in 0..count {
            // Alternate left and right of the leader.
            let idx = (i as f32 + 1.0) / 2.0;
            let side = if i % 2 == 0 { 1.0 } else { -1.0 };
            offsets.push(Vec2::new(side * idx * spacing, 0.0));
        }
        offsets
    }

    /// Generate offsets for a column formation (stacked behind leader).
    fn generate_column_offsets(count: usize, spacing: f32) -> Vec<Vec2> {
        let mut offsets = Vec::with_capacity(count);
        for i in 0..count {
            offsets.push(Vec2::new(0.0, -(i as f32 + 1.0) * spacing));
        }
        offsets
    }

    /// Generate offsets for a wedge/V formation.
    fn generate_wedge_offsets(count: usize, angle_deg: f32, spacing: f32) -> Vec<Vec2> {
        let angle_rad = angle_deg.to_radians();
        let mut offsets = Vec::with_capacity(count);

        for i in 0..count {
            let rank = (i / 2) + 1;
            let side = if i % 2 == 0 { 1.0_f32 } else { -1.0 };

            let along = -(rank as f32) * spacing * angle_rad.cos();
            let lateral = side * (rank as f32) * spacing * angle_rad.sin();

            offsets.push(Vec2::new(lateral, along));
        }
        offsets
    }

    /// Generate offsets for a circle formation.
    fn generate_circle_offsets(count: usize, radius: f32) -> Vec<Vec2> {
        let mut offsets = Vec::with_capacity(count);
        if count == 0 {
            return offsets;
        }
        let step = std::f32::consts::TAU / count as f32;
        for i in 0..count {
            let angle = step * i as f32;
            offsets.push(Vec2::new(angle.cos() * radius, angle.sin() * radius));
        }
        offsets
    }

    /// Generate offsets for a grid formation.
    fn generate_grid_offsets(count: usize, rows: usize, cols: usize, spacing: f32) -> Vec<Vec2> {
        let mut offsets = Vec::with_capacity(count);
        let half_cols = (cols as f32 - 1.0) / 2.0;

        let mut placed = 0;
        for row in 0..rows {
            for col in 0..cols {
                if placed >= count {
                    break;
                }
                let x = (col as f32 - half_cols) * spacing;
                let y = -(row as f32 + 1.0) * spacing;
                offsets.push(Vec2::new(x, y));
                placed += 1;
            }
        }
        offsets
    }
}

// ---------------------------------------------------------------------------
// FormationSlot
// ---------------------------------------------------------------------------

/// A single slot within a formation.
#[derive(Debug, Clone)]
pub struct FormationSlot {
    /// Unique slot index within the formation.
    pub index: usize,
    /// Local-space offset from the leader.
    pub local_offset: Vec2,
    /// World-space computed position (updated each tick).
    pub world_position: Vec2,
    /// Priority of this slot (lower = more important / closer to leader).
    pub priority: u32,
    /// Entity assigned to this slot (None if empty).
    pub assigned_entity: Option<u64>,
    /// Whether this slot is currently occupied by a living unit.
    pub occupied: bool,
}

impl FormationSlot {
    /// Create a new slot.
    pub fn new(index: usize, local_offset: Vec2, priority: u32) -> Self {
        Self {
            index,
            local_offset,
            world_position: Vec2::ZERO,
            priority,
            assigned_entity: None,
            occupied: false,
        }
    }

    /// Returns `true` if this slot has a unit assigned to it.
    pub fn is_assigned(&self) -> bool {
        self.assigned_entity.is_some()
    }

    /// Clear the assignment.
    pub fn clear(&mut self) {
        self.assigned_entity = None;
        self.occupied = false;
    }

    /// Assign an entity to this slot.
    pub fn assign(&mut self, entity_id: u64) {
        self.assigned_entity = Some(entity_id);
        self.occupied = true;
    }

    /// Compute the world-space position from the leader's transform.
    pub fn compute_world_position(&mut self, leader_pos: Vec2, leader_rotation: f32) {
        let rot = Mat2::from_angle(leader_rotation);
        let rotated = rot * self.local_offset;
        self.world_position = leader_pos + rotated;
    }
}

// ---------------------------------------------------------------------------
// Formation
// ---------------------------------------------------------------------------

/// A formation consisting of a shape and assigned units.
#[derive(Debug, Clone)]
pub struct Formation {
    /// Unique formation identifier.
    pub id: u32,
    /// Human-readable name for this formation.
    pub name: String,
    /// The geometric shape.
    pub shape: FormationShape,
    /// Leader entity ID.
    pub leader: u64,
    /// Leader position in XZ plane.
    pub leader_position: Vec2,
    /// Leader rotation (radians, clockwise from +Z in XZ).
    pub leader_rotation: f32,
    /// All slots in the formation.
    pub slots: Vec<FormationSlot>,
    /// Whether the formation is active.
    pub active: bool,
    /// Movement speed of the entire formation (used for path following).
    pub move_speed: f32,
}

impl Formation {
    /// Create a new formation for the given leader and shape, allocating
    /// enough slots for `unit_count` followers.
    pub fn new(
        id: u32,
        name: impl Into<String>,
        leader: u64,
        shape: FormationShape,
        unit_count: usize,
    ) -> Self {
        let clamped_count = unit_count.min(MAX_FORMATION_UNITS);
        let offsets = shape.generate_offsets(clamped_count);
        let slots = offsets
            .iter()
            .enumerate()
            .map(|(i, &offset)| {
                let priority = (offset.length() / DEFAULT_SPACING.max(0.01)).ceil() as u32;
                FormationSlot::new(i, offset, priority)
            })
            .collect();

        Self {
            id,
            name: name.into(),
            shape,
            leader,
            leader_position: Vec2::ZERO,
            leader_rotation: 0.0,
            slots,
            active: true,
            move_speed: 5.0,
        }
    }

    /// Change the formation shape, regenerating slot offsets and reassigning
    /// units to the closest new slots.
    pub fn change_shape(&mut self, new_shape: FormationShape) {
        let unit_count = self.slots.len();
        let offsets = new_shape.generate_offsets(unit_count);

        // Collect currently assigned units and their current world positions.
        let assigned: Vec<(u64, Vec2)> = self
            .slots
            .iter()
            .filter_map(|s| s.assigned_entity.map(|e| (e, s.world_position)))
            .collect();

        // Rebuild slots.
        self.slots = offsets
            .iter()
            .enumerate()
            .map(|(i, &offset)| {
                let priority = (offset.length() / DEFAULT_SPACING.max(0.01)).ceil() as u32;
                let mut slot = FormationSlot::new(i, offset, priority);
                slot.compute_world_position(self.leader_position, self.leader_rotation);
                slot
            })
            .collect();

        self.shape = new_shape;

        // Reassign units to nearest new slots.
        let unit_positions: Vec<(u64, Vec2)> = assigned;
        self.assign_units_to_slots_internal(&unit_positions);
    }

    /// Set the move speed.
    pub fn with_move_speed(mut self, speed: f32) -> Self {
        self.move_speed = speed;
        self
    }

    /// Update the formation's world-space slot positions based on the
    /// leader's current position and rotation.
    pub fn update(&mut self, leader_pos: Vec2, leader_rotation: f32) {
        self.leader_position = leader_pos;
        self.leader_rotation = leader_rotation;

        for slot in &mut self.slots {
            slot.compute_world_position(leader_pos, leader_rotation);
        }
    }

    /// Assign a list of units to formation slots using greedy
    /// nearest-neighbor matching.
    ///
    /// Each unit is assigned to the closest unoccupied slot. Units are
    /// processed in order of distance to the formation center for stability.
    pub fn assign_units_to_slots(&mut self, units: &[(u64, Vec2)]) {
        // Clear existing assignments.
        for slot in &mut self.slots {
            slot.clear();
        }
        self.assign_units_to_slots_internal(units);
    }

    /// Internal greedy nearest-neighbor assignment.
    fn assign_units_to_slots_internal(&mut self, units: &[(u64, Vec2)]) {
        // Sort units by distance to leader for stable assignment.
        let mut unit_list: Vec<(u64, Vec2)> = units.to_vec();
        let center = self.leader_position;
        unit_list.sort_by(|a, b| {
            let da = (a.1 - center).length();
            let db = (b.1 - center).length();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut assigned_slots: Vec<bool> = vec![false; self.slots.len()];

        for (entity_id, unit_pos) in &unit_list {
            let mut best_slot = None;
            let mut best_dist = f32::MAX;

            for (i, slot) in self.slots.iter().enumerate() {
                if assigned_slots[i] {
                    continue;
                }
                let dist = (*unit_pos - slot.world_position).length();
                if dist < best_dist {
                    best_dist = dist;
                    best_slot = Some(i);
                }
            }

            if let Some(si) = best_slot {
                self.slots[si].assign(*entity_id);
                assigned_slots[si] = true;
            }
        }
    }

    /// Handle a unit being removed (death, departure). Clears the slot
    /// and optionally triggers reassignment if `reassign` is true.
    pub fn remove_unit(&mut self, entity_id: u64, reassign: bool) {
        for slot in &mut self.slots {
            if slot.assigned_entity == Some(entity_id) {
                slot.clear();
                break;
            }
        }

        if reassign {
            self.reassign_remaining();
        }
    }

    /// Reassign all remaining units to minimize total distance to slots.
    pub fn reassign_remaining(&mut self) {
        let units: Vec<(u64, Vec2)> = self
            .slots
            .iter()
            .filter_map(|s| s.assigned_entity.map(|e| (e, s.world_position)))
            .collect();

        for slot in &mut self.slots {
            slot.clear();
        }

        self.assign_units_to_slots_internal(&units);
    }

    /// Add new slots for additional units joining the formation.
    pub fn add_units(&mut self, new_units: &[(u64, Vec2)]) {
        let current_count = self.slots.len();
        let new_count = current_count + new_units.len();
        let all_offsets = self.shape.generate_offsets(new_count);

        // Add new slots.
        for i in current_count..new_count.min(all_offsets.len()) {
            let offset = all_offsets[i];
            let priority = (offset.length() / DEFAULT_SPACING.max(0.01)).ceil() as u32;
            let mut slot = FormationSlot::new(i, offset, priority);
            slot.compute_world_position(self.leader_position, self.leader_rotation);
            self.slots.push(slot);
        }

        // Assign the new units to the newly created (empty) slots.
        let empty_slots: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_assigned())
            .map(|(i, _)| i)
            .collect();

        for (j, (entity_id, _unit_pos)) in new_units.iter().enumerate() {
            if j < empty_slots.len() {
                self.slots[empty_slots[j]].assign(*entity_id);
            }
        }
    }

    /// Remove slots, unassigning any entities in them.
    pub fn shrink_to(&mut self, new_count: usize) {
        while self.slots.len() > new_count {
            self.slots.pop();
        }
    }

    /// Get the target position for a specific entity.
    pub fn get_target_position(&self, entity_id: u64) -> Option<Vec2> {
        self.slots
            .iter()
            .find(|s| s.assigned_entity == Some(entity_id))
            .map(|s| s.world_position)
    }

    /// Get all assigned entity IDs.
    pub fn assigned_entities(&self) -> Vec<u64> {
        self.slots
            .iter()
            .filter_map(|s| s.assigned_entity)
            .collect()
    }

    /// Returns the number of occupied slots.
    pub fn occupied_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_assigned()).count()
    }

    /// Returns the total number of slots.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

// ---------------------------------------------------------------------------
// FormationManager
// ---------------------------------------------------------------------------

/// Manages all active formations.
///
/// Provides creation, updates (leader transform tracking), dynamic resize,
/// and slot assignment operations across multiple formations.
///
/// # Usage
///
/// ```ignore
/// let mut mgr = FormationManager::new();
///
/// let fid = mgr.create_formation(
///     "Alpha Squad",
///     leader_entity,
///     FormationShape::wedge(),
///     &[(unit_a, pos_a), (unit_b, pos_b)],
/// );
///
/// // Each tick:
/// mgr.update_formation(fid, leader_pos, leader_rot);
/// for (entity, target) in mgr.get_targets(fid) {
///     // Steer entity toward target.
/// }
/// ```
pub struct FormationManager {
    /// All managed formations, keyed by formation ID.
    formations: HashMap<u32, Formation>,
    /// Next formation ID to assign.
    next_id: u32,
    /// Mapping from entity ID to formation ID (for quick lookup).
    entity_to_formation: HashMap<u64, u32>,
}

impl FormationManager {
    /// Create a new, empty formation manager.
    pub fn new() -> Self {
        Self {
            formations: HashMap::new(),
            next_id: 1,
            entity_to_formation: HashMap::new(),
        }
    }

    /// Create a new formation and return its ID.
    ///
    /// Assigns units to slots using greedy nearest-neighbor matching.
    pub fn create_formation(
        &mut self,
        name: impl Into<String>,
        leader: u64,
        shape: FormationShape,
        units: &[(u64, Vec2)],
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let mut formation = Formation::new(id, name, leader, shape, units.len());
        formation.assign_units_to_slots(units);

        // Track entity -> formation mapping.
        self.entity_to_formation.insert(leader, id);
        for (entity_id, _) in units {
            self.entity_to_formation.insert(*entity_id, id);
        }

        self.formations.insert(id, formation);
        id
    }

    /// Update a formation's leader transform and recompute slot positions.
    pub fn update_formation(
        &mut self,
        formation_id: u32,
        leader_pos: Vec2,
        leader_rotation: f32,
    ) {
        if let Some(formation) = self.formations.get_mut(&formation_id) {
            formation.update(leader_pos, leader_rotation);
        }
    }

    /// Get the target positions for all assigned units in a formation.
    pub fn get_targets(&self, formation_id: u32) -> Vec<(u64, Vec2)> {
        self.formations
            .get(&formation_id)
            .map(|f| {
                f.slots
                    .iter()
                    .filter_map(|s| s.assigned_entity.map(|e| (e, s.world_position)))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remove a unit from its formation.
    pub fn remove_unit(&mut self, entity_id: u64, reassign: bool) {
        if let Some(&fid) = self.entity_to_formation.get(&entity_id) {
            if let Some(formation) = self.formations.get_mut(&fid) {
                formation.remove_unit(entity_id, reassign);
            }
            self.entity_to_formation.remove(&entity_id);
        }
    }

    /// Add units to an existing formation.
    pub fn add_units_to_formation(
        &mut self,
        formation_id: u32,
        units: &[(u64, Vec2)],
    ) {
        if let Some(formation) = self.formations.get_mut(&formation_id) {
            formation.add_units(units);
            for (entity_id, _) in units {
                self.entity_to_formation.insert(*entity_id, formation_id);
            }
        }
    }

    /// Change the shape of an existing formation.
    pub fn change_formation_shape(&mut self, formation_id: u32, new_shape: FormationShape) {
        if let Some(formation) = self.formations.get_mut(&formation_id) {
            formation.change_shape(new_shape);
        }
    }

    /// Dissolve a formation entirely.
    pub fn dissolve_formation(&mut self, formation_id: u32) {
        if let Some(formation) = self.formations.remove(&formation_id) {
            self.entity_to_formation.remove(&formation.leader);
            for slot in &formation.slots {
                if let Some(eid) = slot.assigned_entity {
                    self.entity_to_formation.remove(&eid);
                }
            }
        }
    }

    /// Get a reference to a formation.
    pub fn get_formation(&self, formation_id: u32) -> Option<&Formation> {
        self.formations.get(&formation_id)
    }

    /// Get a mutable reference to a formation.
    pub fn get_formation_mut(&mut self, formation_id: u32) -> Option<&mut Formation> {
        self.formations.get_mut(&formation_id)
    }

    /// Look up which formation an entity belongs to.
    pub fn entity_formation(&self, entity_id: u64) -> Option<u32> {
        self.entity_to_formation.get(&entity_id).copied()
    }

    /// Returns the number of active formations.
    pub fn formation_count(&self) -> usize {
        self.formations.len()
    }

    /// Clear all formations.
    pub fn clear(&mut self) {
        self.formations.clear();
        self.entity_to_formation.clear();
    }
}

impl Default for FormationManager {
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
    fn test_line_offsets() {
        let shape = FormationShape::Line { spacing: 2.0 };
        let offsets = shape.generate_offsets(4);
        assert_eq!(offsets.len(), 4);
        // First unit should be to the right, second to the left.
        assert!(offsets[0].x > 0.0);
        assert!(offsets[1].x < 0.0);
    }

    #[test]
    fn test_column_offsets() {
        let shape = FormationShape::Column { spacing: 3.0 };
        let offsets = shape.generate_offsets(3);
        assert_eq!(offsets.len(), 3);
        // All units should be behind the leader (negative Y).
        for offset in &offsets {
            assert!(offset.y < 0.0);
            assert!((offset.x).abs() < EPSILON);
        }
    }

    #[test]
    fn test_wedge_offsets() {
        let shape = FormationShape::Wedge {
            angle: 45.0,
            spacing: 2.0,
        };
        let offsets = shape.generate_offsets(4);
        assert_eq!(offsets.len(), 4);
        // Alternating sides.
        assert!(offsets[0].x > 0.0); // Right arm.
        assert!(offsets[1].x < 0.0); // Left arm.
    }

    #[test]
    fn test_circle_offsets() {
        let shape = FormationShape::Circle { radius: 5.0 };
        let offsets = shape.generate_offsets(4);
        assert_eq!(offsets.len(), 4);
        // All offsets should be at the same distance from origin.
        for offset in &offsets {
            let dist = offset.length();
            assert!((dist - 5.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_grid_offsets() {
        let shape = FormationShape::Grid {
            rows: 2,
            cols: 3,
            spacing: 2.0,
        };
        let offsets = shape.generate_offsets(6);
        assert_eq!(offsets.len(), 6);
        // All behind leader.
        for offset in &offsets {
            assert!(offset.y < 0.0);
        }
    }

    #[test]
    fn test_custom_offsets() {
        let custom = vec![Vec2::new(1.0, 0.0), Vec2::new(-1.0, 0.0)];
        let shape = FormationShape::Custom { offsets: custom };
        let offsets = shape.generate_offsets(2);
        assert_eq!(offsets.len(), 2);
        assert_eq!(offsets[0], Vec2::new(1.0, 0.0));
    }

    #[test]
    fn test_formation_slot_world_position() {
        let mut slot = FormationSlot::new(0, Vec2::new(2.0, 0.0), 1);
        slot.compute_world_position(Vec2::new(10.0, 10.0), 0.0);

        assert!((slot.world_position.x - 12.0).abs() < 0.01);
        assert!((slot.world_position.y - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_formation_slot_rotated() {
        let mut slot = FormationSlot::new(0, Vec2::new(2.0, 0.0), 1);
        let rot = std::f32::consts::FRAC_PI_2; // 90 degrees.
        slot.compute_world_position(Vec2::ZERO, rot);

        // After 90-degree rotation, (2,0) becomes (0,2).
        assert!(slot.world_position.x.abs() < 0.01);
        assert!((slot.world_position.y - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_formation_create_and_assign() {
        let mut formation = Formation::new(1, "Test", 100, FormationShape::line(), 3);
        formation.update(Vec2::new(5.0, 5.0), 0.0);

        let units = vec![
            (201, Vec2::new(6.0, 5.0)),
            (202, Vec2::new(4.0, 5.0)),
            (203, Vec2::new(7.0, 5.0)),
        ];

        formation.assign_units_to_slots(&units);
        assert_eq!(formation.occupied_count(), 3);
        assert!(formation.get_target_position(201).is_some());
    }

    #[test]
    fn test_formation_remove_unit() {
        let mut formation = Formation::new(1, "Test", 100, FormationShape::column(), 2);
        formation.update(Vec2::ZERO, 0.0);

        let units = vec![(201, Vec2::ZERO), (202, Vec2::new(0.0, -2.0))];
        formation.assign_units_to_slots(&units);
        assert_eq!(formation.occupied_count(), 2);

        formation.remove_unit(201, false);
        assert_eq!(formation.occupied_count(), 1);
    }

    #[test]
    fn test_formation_remove_and_reassign() {
        let mut formation = Formation::new(1, "Test", 100, FormationShape::line(), 3);
        formation.update(Vec2::ZERO, 0.0);

        let units = vec![
            (201, Vec2::new(2.0, 0.0)),
            (202, Vec2::new(-2.0, 0.0)),
            (203, Vec2::new(4.0, 0.0)),
        ];
        formation.assign_units_to_slots(&units);

        formation.remove_unit(201, true);
        assert_eq!(formation.occupied_count(), 2);
    }

    #[test]
    fn test_formation_add_units() {
        let mut formation = Formation::new(1, "Test", 100, FormationShape::line(), 2);
        formation.update(Vec2::ZERO, 0.0);

        let initial = vec![(201, Vec2::ZERO), (202, Vec2::ZERO)];
        formation.assign_units_to_slots(&initial);
        assert_eq!(formation.slot_count(), 2);

        formation.add_units(&[(203, Vec2::ZERO)]);
        assert_eq!(formation.slot_count(), 3);
        assert_eq!(formation.occupied_count(), 3);
    }

    #[test]
    fn test_formation_change_shape() {
        let mut formation = Formation::new(1, "Test", 100, FormationShape::line(), 3);
        formation.update(Vec2::ZERO, 0.0);

        let units = vec![
            (201, Vec2::new(2.0, 0.0)),
            (202, Vec2::new(-2.0, 0.0)),
            (203, Vec2::new(4.0, 0.0)),
        ];
        formation.assign_units_to_slots(&units);

        formation.change_shape(FormationShape::column());
        assert_eq!(formation.occupied_count(), 3);
    }

    #[test]
    fn test_formation_manager_lifecycle() {
        let mut mgr = FormationManager::new();

        let units = vec![(10, Vec2::new(1.0, 0.0)), (11, Vec2::new(-1.0, 0.0))];
        let fid = mgr.create_formation("Squad A", 1, FormationShape::wedge(), &units);

        assert_eq!(mgr.formation_count(), 1);
        assert_eq!(mgr.entity_formation(10), Some(fid));

        mgr.update_formation(fid, Vec2::new(5.0, 5.0), 0.0);

        let targets = mgr.get_targets(fid);
        assert_eq!(targets.len(), 2);

        mgr.remove_unit(10, true);
        assert_eq!(mgr.entity_formation(10), None);

        mgr.dissolve_formation(fid);
        assert_eq!(mgr.formation_count(), 0);
    }

    #[test]
    fn test_greedy_assignment_minimizes_distance() {
        let mut formation = Formation::new(1, "Test", 100, FormationShape::line(), 2);
        formation.update(Vec2::ZERO, 0.0);

        // Unit A is close to slot on the right, Unit B is close to slot on left.
        let units = vec![
            (201, Vec2::new(3.0, 0.0)),   // Near right slot.
            (202, Vec2::new(-3.0, 0.0)),  // Near left slot.
        ];
        formation.assign_units_to_slots(&units);

        // The closer unit should have been assigned to the closer slot.
        let pos_a = formation.get_target_position(201).unwrap();
        let pos_b = formation.get_target_position(202).unwrap();
        assert!(pos_a.x > 0.0); // Right slot.
        assert!(pos_b.x < 0.0); // Left slot.
    }
}
