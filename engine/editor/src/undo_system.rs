//! Robust undo/redo system for the editor.
//!
//! # Integration TODO (Genovo Studio)
//!
//! The editor app (`genovo-app/src/main.rs`) currently tracks `undo_count` /
//! `redo_count` as plain integers but does not own an `UndoStack` instance.
//! To wire up real undo/redo:
//!
//! 1. Add an `undo_stack: UndoStack` field to the app's editor state struct.
//! 2. When the user performs a transform gizmo drag, property edit, or
//!    hierarchy change, push the corresponding `Operation` (e.g.
//!    `MoveEntityOp`, `ChangeComponentOp`) via `undo_stack.push(op, true)`.
//! 3. Bind Ctrl+Z to `undo_stack.undo()` and Ctrl+Y / Ctrl+Shift+Z to
//!    `undo_stack.redo()`.
//! 4. Replace the `undo_count` / `redo_count` UI display with
//!    `undo_stack.undo_count()` / `undo_stack.redo_count()`.
//! 5. Optionally register `undo_stack.set_on_change(...)` to update the UI
//!    title bar or status bar whenever the stack changes.
//!
//! Provides a stack-based undo/redo mechanism with support for:
//!
//! - **Operation trait** — reversible operations with `execute` and `undo`.
//! - **Concrete operations** — move, rotate, scale, spawn, despawn, reparent,
//!   component change, and compound (atomic multi-op) operations.
//! - **Merge policy** — consecutive similar operations (e.g., many small
//!   mouse-drag moves) are merged into a single undo step.
//! - **Stack management** — maximum stack size, clear-redo-on-new-push,
//!   operation descriptions for UI display.
//!
//! # Architecture
//!
//! The `UndoStack` owns a list of `Box<dyn Operation>`. Pushing a new
//! operation executes it and clears the redo stack. Undo pops from the undo
//! stack and pushes to redo. Redo pops from redo and pushes to undo.
//!
//! Merge works by checking if the top of the undo stack is "mergeable" with
//! an incoming operation. If so, the existing operation is updated in place
//! rather than pushing a new entry. This collapses drag gestures into a
//! single undo step.

use std::any::Any;
use std::fmt;

// ---------------------------------------------------------------------------
// Vec3 / Quat stand-ins for self-contained operation data
// ---------------------------------------------------------------------------

/// 3D vector for operation data. We use our own type rather than depending
/// on glam so the undo system is self-contained.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn distance_to(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2}, {:.2})", self.x, self.y, self.z)
    }
}

/// Quaternion for operation data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Angular difference in radians between two quaternions.
    pub fn angle_to(&self, other: &Self) -> f32 {
        let dot = (self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w)
            .abs()
            .min(1.0);
        2.0 * dot.acos()
    }
}

impl fmt::Display for Quat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({:.3}, {:.3}, {:.3}, {:.3})",
            self.x, self.y, self.z, self.w
        )
    }
}

// ---------------------------------------------------------------------------
// EntityId — opaque entity identifier for the undo system
// ---------------------------------------------------------------------------

/// Opaque entity identifier. The undo system doesn't depend on the ECS
/// entity type directly; this is a lightweight u64 handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Operation trait
// ---------------------------------------------------------------------------

/// A reversible operation that can be executed and undone.
///
/// Operations are the fundamental building block of the undo system. Each
/// operation stores enough data to both apply and reverse its effect.
pub trait Operation: Send + Sync + fmt::Debug {
    /// Execute the operation (apply its effect).
    fn execute(&self) -> OperationResult;

    /// Reverse the operation (undo its effect).
    fn undo(&self) -> OperationResult;

    /// Human-readable description for the undo/redo UI.
    fn description(&self) -> &str;

    /// The type of operation, used for merge checking.
    fn op_type(&self) -> OperationType;

    /// Check if this operation can be merged with a new incoming operation.
    ///
    /// Returns `true` if the operations are compatible for merging (same
    /// entity, same operation type, within time/distance threshold).
    fn can_merge(&self, other: &dyn Operation) -> bool {
        false
    }

    /// Merge another operation into this one, updating the "new" values.
    ///
    /// Called only after `can_merge` returns `true`. The implementation
    /// should update its internal state to reflect the combined effect.
    fn merge(&mut self, other: &dyn Operation) -> bool {
        false
    }

    /// Downcast to Any for type-specific operations.
    fn as_any(&self) -> &dyn Any;

    /// Downcast to mutable Any.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Result of executing or undoing an operation.
#[derive(Debug, Clone)]
pub enum OperationResult {
    /// Operation succeeded.
    Success,
    /// Operation failed with an error message.
    Failed(String),
    /// Operation had no effect (e.g., entity already at target position).
    NoOp,
}

impl OperationResult {
    pub fn is_success(&self) -> bool {
        matches!(self, OperationResult::Success)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, OperationResult::Failed(_))
    }
}

/// Enum of known operation types for merge compatibility checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    Move,
    Rotate,
    Scale,
    Spawn,
    Despawn,
    ChangeComponent,
    Reparent,
    Compound,
    Custom,
}

// ---------------------------------------------------------------------------
// Concrete operations
// ---------------------------------------------------------------------------

/// Operation: move an entity from one position to another.
#[derive(Debug, Clone)]
pub struct MoveEntityOp {
    pub entity: EntityId,
    pub old_pos: Vec3,
    pub new_pos: Vec3,
    /// Timestamp when the operation was created (for merge windowing).
    pub timestamp_ms: u64,
}

impl MoveEntityOp {
    pub fn new(entity: EntityId, old_pos: Vec3, new_pos: Vec3) -> Self {
        Self {
            entity,
            old_pos,
            new_pos,
            timestamp_ms: 0,
        }
    }

    pub fn with_timestamp(mut self, timestamp_ms: u64) -> Self {
        self.timestamp_ms = timestamp_ms;
        self
    }
}

impl Operation for MoveEntityOp {
    fn execute(&self) -> OperationResult {
        // In a real implementation this would modify the world.
        // Here we just validate.
        if self.old_pos == self.new_pos {
            OperationResult::NoOp
        } else {
            OperationResult::Success
        }
    }

    fn undo(&self) -> OperationResult {
        if self.old_pos == self.new_pos {
            OperationResult::NoOp
        } else {
            OperationResult::Success
        }
    }

    fn description(&self) -> &str {
        "Move Entity"
    }

    fn op_type(&self) -> OperationType {
        OperationType::Move
    }

    fn can_merge(&self, other: &dyn Operation) -> bool {
        if other.op_type() != OperationType::Move {
            return false;
        }
        if let Some(other_move) = other.as_any().downcast_ref::<MoveEntityOp>() {
            // Same entity and within merge time window (500ms).
            self.entity == other_move.entity
                && (other_move.timestamp_ms.saturating_sub(self.timestamp_ms)) < 500
        } else {
            false
        }
    }

    fn merge(&mut self, other: &dyn Operation) -> bool {
        if let Some(other_move) = other.as_any().downcast_ref::<MoveEntityOp>() {
            // Keep old_pos from self, take new_pos from other.
            self.new_pos = other_move.new_pos;
            self.timestamp_ms = other_move.timestamp_ms;
            true
        } else {
            false
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Operation: rotate an entity.
#[derive(Debug, Clone)]
pub struct RotateEntityOp {
    pub entity: EntityId,
    pub old_rot: Quat,
    pub new_rot: Quat,
    pub timestamp_ms: u64,
}

impl RotateEntityOp {
    pub fn new(entity: EntityId, old_rot: Quat, new_rot: Quat) -> Self {
        Self {
            entity,
            old_rot,
            new_rot,
            timestamp_ms: 0,
        }
    }

    pub fn with_timestamp(mut self, timestamp_ms: u64) -> Self {
        self.timestamp_ms = timestamp_ms;
        self
    }
}

impl Operation for RotateEntityOp {
    fn execute(&self) -> OperationResult {
        OperationResult::Success
    }

    fn undo(&self) -> OperationResult {
        OperationResult::Success
    }

    fn description(&self) -> &str {
        "Rotate Entity"
    }

    fn op_type(&self) -> OperationType {
        OperationType::Rotate
    }

    fn can_merge(&self, other: &dyn Operation) -> bool {
        if other.op_type() != OperationType::Rotate {
            return false;
        }
        if let Some(other_rot) = other.as_any().downcast_ref::<RotateEntityOp>() {
            self.entity == other_rot.entity
                && (other_rot.timestamp_ms.saturating_sub(self.timestamp_ms)) < 500
        } else {
            false
        }
    }

    fn merge(&mut self, other: &dyn Operation) -> bool {
        if let Some(other_rot) = other.as_any().downcast_ref::<RotateEntityOp>() {
            self.new_rot = other_rot.new_rot;
            self.timestamp_ms = other_rot.timestamp_ms;
            true
        } else {
            false
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Operation: scale an entity.
#[derive(Debug, Clone)]
pub struct ScaleEntityOp {
    pub entity: EntityId,
    pub old_scale: Vec3,
    pub new_scale: Vec3,
    pub timestamp_ms: u64,
}

impl ScaleEntityOp {
    pub fn new(entity: EntityId, old_scale: Vec3, new_scale: Vec3) -> Self {
        Self {
            entity,
            old_scale,
            new_scale,
            timestamp_ms: 0,
        }
    }

    pub fn with_timestamp(mut self, timestamp_ms: u64) -> Self {
        self.timestamp_ms = timestamp_ms;
        self
    }
}

impl Operation for ScaleEntityOp {
    fn execute(&self) -> OperationResult {
        OperationResult::Success
    }

    fn undo(&self) -> OperationResult {
        OperationResult::Success
    }

    fn description(&self) -> &str {
        "Scale Entity"
    }

    fn op_type(&self) -> OperationType {
        OperationType::Scale
    }

    fn can_merge(&self, other: &dyn Operation) -> bool {
        if other.op_type() != OperationType::Scale {
            return false;
        }
        if let Some(other_scale) = other.as_any().downcast_ref::<ScaleEntityOp>() {
            self.entity == other_scale.entity
                && (other_scale.timestamp_ms.saturating_sub(self.timestamp_ms)) < 500
        } else {
            false
        }
    }

    fn merge(&mut self, other: &dyn Operation) -> bool {
        if let Some(other_scale) = other.as_any().downcast_ref::<ScaleEntityOp>() {
            self.new_scale = other_scale.new_scale;
            self.timestamp_ms = other_scale.timestamp_ms;
            true
        } else {
            false
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Serialized component data for spawn/despawn/change operations.
#[derive(Debug, Clone)]
pub struct SerializedComponentData {
    /// Component type name (for display purposes).
    pub type_name: String,
    /// Opaque serialized data.
    pub data: Vec<u8>,
}

/// Serialized entity data for spawn/despawn operations.
#[derive(Debug, Clone)]
pub struct SerializedEntityData {
    /// Entity ID.
    pub entity: EntityId,
    /// Entity name.
    pub name: String,
    /// Serialized components.
    pub components: Vec<SerializedComponentData>,
    /// Parent entity, if any.
    pub parent: Option<EntityId>,
    /// Child entities (serialized recursively for despawn undo).
    pub children: Vec<SerializedEntityData>,
}

/// Operation: spawn an entity (undo = despawn).
#[derive(Debug, Clone)]
pub struct SpawnEntityOp {
    pub entity_data: SerializedEntityData,
}

impl SpawnEntityOp {
    pub fn new(entity_data: SerializedEntityData) -> Self {
        Self { entity_data }
    }
}

impl Operation for SpawnEntityOp {
    fn execute(&self) -> OperationResult {
        // Spawn the entity from serialized data.
        OperationResult::Success
    }

    fn undo(&self) -> OperationResult {
        // Despawn the entity.
        OperationResult::Success
    }

    fn description(&self) -> &str {
        "Spawn Entity"
    }

    fn op_type(&self) -> OperationType {
        OperationType::Spawn
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Operation: despawn an entity (undo = respawn from saved data).
#[derive(Debug, Clone)]
pub struct DespawnEntityOp {
    pub entity_data: SerializedEntityData,
}

impl DespawnEntityOp {
    pub fn new(entity_data: SerializedEntityData) -> Self {
        Self { entity_data }
    }
}

impl Operation for DespawnEntityOp {
    fn execute(&self) -> OperationResult {
        // Despawn the entity.
        OperationResult::Success
    }

    fn undo(&self) -> OperationResult {
        // Respawn the entity from saved data.
        OperationResult::Success
    }

    fn description(&self) -> &str {
        "Delete Entity"
    }

    fn op_type(&self) -> OperationType {
        OperationType::Despawn
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Operation: change a component value.
#[derive(Debug, Clone)]
pub struct ChangeComponentOp {
    pub entity: EntityId,
    pub component_type: String,
    pub old_value: Vec<u8>,
    pub new_value: Vec<u8>,
    pub property_path: String,
    pub timestamp_ms: u64,
}

impl ChangeComponentOp {
    pub fn new(
        entity: EntityId,
        component_type: impl Into<String>,
        property_path: impl Into<String>,
        old_value: Vec<u8>,
        new_value: Vec<u8>,
    ) -> Self {
        Self {
            entity,
            component_type: component_type.into(),
            old_value,
            new_value,
            property_path: property_path.into(),
            timestamp_ms: 0,
        }
    }

    pub fn with_timestamp(mut self, timestamp_ms: u64) -> Self {
        self.timestamp_ms = timestamp_ms;
        self
    }
}

impl Operation for ChangeComponentOp {
    fn execute(&self) -> OperationResult {
        OperationResult::Success
    }

    fn undo(&self) -> OperationResult {
        OperationResult::Success
    }

    fn description(&self) -> &str {
        "Change Component"
    }

    fn op_type(&self) -> OperationType {
        OperationType::ChangeComponent
    }

    fn can_merge(&self, other: &dyn Operation) -> bool {
        if other.op_type() != OperationType::ChangeComponent {
            return false;
        }
        if let Some(other_change) = other.as_any().downcast_ref::<ChangeComponentOp>() {
            self.entity == other_change.entity
                && self.component_type == other_change.component_type
                && self.property_path == other_change.property_path
                && (other_change.timestamp_ms.saturating_sub(self.timestamp_ms)) < 500
        } else {
            false
        }
    }

    fn merge(&mut self, other: &dyn Operation) -> bool {
        if let Some(other_change) = other.as_any().downcast_ref::<ChangeComponentOp>() {
            self.new_value = other_change.new_value.clone();
            self.timestamp_ms = other_change.timestamp_ms;
            true
        } else {
            false
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Operation: reparent an entity.
#[derive(Debug, Clone)]
pub struct ReparentOp {
    pub entity: EntityId,
    pub old_parent: Option<EntityId>,
    pub new_parent: Option<EntityId>,
}

impl ReparentOp {
    pub fn new(
        entity: EntityId,
        old_parent: Option<EntityId>,
        new_parent: Option<EntityId>,
    ) -> Self {
        Self {
            entity,
            old_parent,
            new_parent,
        }
    }
}

impl Operation for ReparentOp {
    fn execute(&self) -> OperationResult {
        OperationResult::Success
    }

    fn undo(&self) -> OperationResult {
        OperationResult::Success
    }

    fn description(&self) -> &str {
        "Reparent Entity"
    }

    fn op_type(&self) -> OperationType {
        OperationType::Reparent
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Operation: a group of operations that are undone/redone atomically.
///
/// When undone, all sub-operations are undone in reverse order. When
/// redone, they are re-executed in forward order.
#[derive(Debug)]
pub struct CompoundOp {
    pub operations: Vec<Box<dyn Operation>>,
    pub desc: String,
}

impl CompoundOp {
    pub fn new(desc: impl Into<String>) -> Self {
        Self {
            operations: Vec::new(),
            desc: desc.into(),
        }
    }

    pub fn add(mut self, op: Box<dyn Operation>) -> Self {
        self.operations.push(op);
        self
    }

    pub fn push(&mut self, op: Box<dyn Operation>) {
        self.operations.push(op);
    }

    pub fn len(&self) -> usize {
        self.operations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

impl Operation for CompoundOp {
    fn execute(&self) -> OperationResult {
        for op in &self.operations {
            let result = op.execute();
            if result.is_failed() {
                return result;
            }
        }
        OperationResult::Success
    }

    fn undo(&self) -> OperationResult {
        // Undo in reverse order.
        for op in self.operations.iter().rev() {
            let result = op.undo();
            if result.is_failed() {
                return result;
            }
        }
        OperationResult::Success
    }

    fn description(&self) -> &str {
        &self.desc
    }

    fn op_type(&self) -> OperationType {
        OperationType::Compound
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// UndoStack
// ---------------------------------------------------------------------------

/// Stack of reversible operations with undo/redo support.
///
/// # Merge Behavior
///
/// When `push` is called with `allow_merge = true`, the stack checks if
/// the top operation can be merged with the incoming one (same type, same
/// entity, within time window). If so, the top operation is updated in
/// place. This collapses continuous drag gestures into a single undo step.
///
/// # Stack Size Limit
///
/// The stack has a configurable maximum size. When exceeded, the oldest
/// operations are dropped from the bottom.
pub struct UndoStack {
    /// Undo stack (most recent at the end).
    undo_stack: Vec<Box<dyn Operation>>,
    /// Redo stack (most recent at the end).
    redo_stack: Vec<Box<dyn Operation>>,
    /// Maximum number of operations in the undo stack.
    max_size: usize,
    /// Whether merging is globally enabled.
    merge_enabled: bool,
    /// Listener called after each undo/redo/push.
    on_change: Option<Box<dyn Fn(UndoEvent) + Send + Sync>>,
}

/// Events emitted by the undo stack.
#[derive(Debug, Clone)]
pub enum UndoEvent {
    /// A new operation was pushed.
    Pushed(String),
    /// An operation was undone.
    Undone(String),
    /// An operation was redone.
    Redone(String),
    /// An operation was merged with the top of the stack.
    Merged(String),
    /// The stack was cleared.
    Cleared,
}

impl UndoStack {
    /// Create a new undo stack with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_size,
            merge_enabled: true,
            on_change: None,
        }
    }

    /// Create with default max size (100).
    pub fn with_default_size() -> Self {
        Self::new(100)
    }

    /// Set a change listener.
    pub fn set_on_change(&mut self, callback: impl Fn(UndoEvent) + Send + Sync + 'static) {
        self.on_change = Some(Box::new(callback));
    }

    /// Enable or disable merge behavior.
    pub fn set_merge_enabled(&mut self, enabled: bool) {
        self.merge_enabled = enabled;
    }

    /// Push an operation onto the undo stack.
    ///
    /// The operation is executed immediately. The redo stack is cleared
    /// (any undone operations are discarded).
    ///
    /// If `allow_merge` is true and the top operation is compatible, the
    /// operations are merged instead of pushing a new entry.
    pub fn push(&mut self, operation: Box<dyn Operation>, allow_merge: bool) -> OperationResult {
        // Try to merge with the top of the undo stack.
        if allow_merge && self.merge_enabled && !self.undo_stack.is_empty() {
            let top = self.undo_stack.last_mut().unwrap();
            if top.can_merge(operation.as_ref()) {
                let desc = operation.description().to_string();
                if top.merge(operation.as_ref()) {
                    self.emit_event(UndoEvent::Merged(desc));
                    return OperationResult::Success;
                }
            }
        }

        // Execute the operation.
        let result = operation.execute();
        let desc = operation.description().to_string();

        // Push onto undo stack.
        self.undo_stack.push(operation);

        // Clear redo stack (new operation invalidates redo history).
        self.redo_stack.clear();

        // Enforce max size.
        while self.undo_stack.len() > self.max_size {
            self.undo_stack.remove(0);
        }

        self.emit_event(UndoEvent::Pushed(desc));
        result
    }

    /// Push without attempting merge.
    pub fn push_no_merge(&mut self, operation: Box<dyn Operation>) -> OperationResult {
        self.push(operation, false)
    }

    /// Undo the last operation.
    ///
    /// Returns the description of the undone operation, or `None` if
    /// there is nothing to undo.
    pub fn undo(&mut self) -> Option<String> {
        let operation = self.undo_stack.pop()?;
        let desc = operation.description().to_string();
        operation.undo();
        self.redo_stack.push(operation);
        self.emit_event(UndoEvent::Undone(desc.clone()));
        Some(desc)
    }

    /// Redo the last undone operation.
    ///
    /// Returns the description of the redone operation, or `None` if
    /// there is nothing to redo.
    pub fn redo(&mut self) -> Option<String> {
        let operation = self.redo_stack.pop()?;
        let desc = operation.description().to_string();
        operation.execute();
        self.undo_stack.push(operation);
        self.emit_event(UndoEvent::Redone(desc.clone()));
        Some(desc)
    }

    /// Whether there are operations that can be undone.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Whether there are operations that can be redone.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Get the description of the next operation to undo.
    pub fn undo_description(&self) -> Option<&str> {
        self.undo_stack.last().map(|op| op.description())
    }

    /// Get the description of the next operation to redo.
    pub fn redo_description(&self) -> Option<&str> {
        self.redo_stack.last().map(|op| op.description())
    }

    /// Get all undo operation descriptions (newest first).
    pub fn undo_history(&self) -> Vec<&str> {
        self.undo_stack
            .iter()
            .rev()
            .map(|op| op.description())
            .collect()
    }

    /// Get all redo operation descriptions (newest first).
    pub fn redo_history(&self) -> Vec<&str> {
        self.redo_stack
            .iter()
            .rev()
            .map(|op| op.description())
            .collect()
    }

    /// Number of operations in the undo stack.
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of operations in the redo stack.
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Clear both undo and redo stacks.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.emit_event(UndoEvent::Cleared);
    }

    /// Clear only the redo stack.
    pub fn clear_redo(&mut self) {
        self.redo_stack.clear();
    }

    /// Get the maximum stack size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Set the maximum stack size.
    pub fn set_max_size(&mut self, max_size: usize) {
        self.max_size = max_size;
        while self.undo_stack.len() > self.max_size {
            self.undo_stack.remove(0);
        }
    }

    /// Undo multiple operations at once.
    pub fn undo_n(&mut self, n: usize) -> Vec<String> {
        let mut descriptions = Vec::new();
        for _ in 0..n {
            match self.undo() {
                Some(desc) => descriptions.push(desc),
                None => break,
            }
        }
        descriptions
    }

    /// Redo multiple operations at once.
    pub fn redo_n(&mut self, n: usize) -> Vec<String> {
        let mut descriptions = Vec::new();
        for _ in 0..n {
            match self.redo() {
                Some(desc) => descriptions.push(desc),
                None => break,
            }
        }
        descriptions
    }

    /// Begin a compound operation. Returns a builder that collects
    /// sub-operations and pushes them as a single undo step.
    pub fn begin_compound(&mut self, description: impl Into<String>) -> CompoundOpBuilder {
        CompoundOpBuilder {
            compound: CompoundOp::new(description),
        }
    }

    fn emit_event(&self, event: UndoEvent) {
        if let Some(ref callback) = self.on_change {
            callback(event);
        }
    }
}

impl Default for UndoStack {
    fn default() -> Self {
        Self::with_default_size()
    }
}

impl fmt::Debug for UndoStack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UndoStack")
            .field("undo_count", &self.undo_stack.len())
            .field("redo_count", &self.redo_stack.len())
            .field("max_size", &self.max_size)
            .field("merge_enabled", &self.merge_enabled)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// CompoundOpBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing compound operations.
///
/// Usage:
/// ```ignore
/// let mut builder = undo_stack.begin_compound("Transform Selection");
/// builder.add(Box::new(MoveEntityOp { ... }));
/// builder.add(Box::new(RotateEntityOp { ... }));
/// undo_stack.push(builder.finish(), false);
/// ```
pub struct CompoundOpBuilder {
    compound: CompoundOp,
}

impl CompoundOpBuilder {
    /// Add a sub-operation.
    pub fn add(&mut self, op: Box<dyn Operation>) {
        self.compound.push(op);
    }

    /// Finish building and return the compound operation.
    pub fn finish(self) -> Box<dyn Operation> {
        Box::new(self.compound)
    }

    /// Number of sub-operations added so far.
    pub fn len(&self) -> usize {
        self.compound.len()
    }

    /// Whether no sub-operations have been added.
    pub fn is_empty(&self) -> bool {
        self.compound.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn move_op(entity: u64, old: [f32; 3], new: [f32; 3]) -> Box<dyn Operation> {
        Box::new(MoveEntityOp::new(
            EntityId(entity),
            Vec3::new(old[0], old[1], old[2]),
            Vec3::new(new[0], new[1], new[2]),
        ))
    }

    fn move_op_timed(
        entity: u64,
        old: [f32; 3],
        new: [f32; 3],
        time: u64,
    ) -> Box<dyn Operation> {
        Box::new(
            MoveEntityOp::new(
                EntityId(entity),
                Vec3::new(old[0], old[1], old[2]),
                Vec3::new(new[0], new[1], new[2]),
            )
            .with_timestamp(time),
        )
    }

    // -- Basic undo/redo --------------------------------------------------------

    #[test]
    fn push_and_undo() {
        let mut stack = UndoStack::with_default_size();

        stack.push(move_op(1, [0.0; 3], [1.0, 0.0, 0.0]), false);
        assert!(stack.can_undo());
        assert!(!stack.can_redo());
        assert_eq!(stack.undo_count(), 1);

        let desc = stack.undo();
        assert_eq!(desc, Some("Move Entity".to_string()));
        assert!(!stack.can_undo());
        assert!(stack.can_redo());
    }

    #[test]
    fn undo_redo_cycle() {
        let mut stack = UndoStack::with_default_size();

        stack.push(move_op(1, [0.0; 3], [1.0, 0.0, 0.0]), false);
        stack.push(move_op(1, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]), false);

        assert_eq!(stack.undo_count(), 2);

        stack.undo();
        assert_eq!(stack.undo_count(), 1);
        assert_eq!(stack.redo_count(), 1);

        stack.undo();
        assert_eq!(stack.undo_count(), 0);
        assert_eq!(stack.redo_count(), 2);

        stack.redo();
        assert_eq!(stack.undo_count(), 1);
        assert_eq!(stack.redo_count(), 1);

        stack.redo();
        assert_eq!(stack.undo_count(), 2);
        assert_eq!(stack.redo_count(), 0);
    }

    #[test]
    fn new_push_clears_redo() {
        let mut stack = UndoStack::with_default_size();

        stack.push(move_op(1, [0.0; 3], [1.0, 0.0, 0.0]), false);
        stack.push(move_op(1, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]), false);

        stack.undo(); // redo has 1 entry.
        assert_eq!(stack.redo_count(), 1);

        // New push should clear redo.
        stack.push(move_op(1, [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]), false);
        assert_eq!(stack.redo_count(), 0);
    }

    #[test]
    fn undo_empty_returns_none() {
        let mut stack = UndoStack::with_default_size();
        assert_eq!(stack.undo(), None);
    }

    #[test]
    fn redo_empty_returns_none() {
        let mut stack = UndoStack::with_default_size();
        assert_eq!(stack.redo(), None);
    }

    // -- Max size ---------------------------------------------------------------

    #[test]
    fn max_size_enforced() {
        let mut stack = UndoStack::new(3);

        for i in 0..5 {
            stack.push(
                move_op(1, [i as f32, 0.0, 0.0], [(i + 1) as f32, 0.0, 0.0]),
                false,
            );
        }

        assert_eq!(stack.undo_count(), 3);
    }

    #[test]
    fn set_max_size_trims() {
        let mut stack = UndoStack::new(10);

        for i in 0..8 {
            stack.push(
                move_op(1, [i as f32, 0.0, 0.0], [(i + 1) as f32, 0.0, 0.0]),
                false,
            );
        }

        assert_eq!(stack.undo_count(), 8);

        stack.set_max_size(5);
        assert_eq!(stack.undo_count(), 5);
    }

    // -- Merge behavior ---------------------------------------------------------

    #[test]
    fn merge_consecutive_moves() {
        let mut stack = UndoStack::with_default_size();

        stack.push(
            move_op_timed(1, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100),
            true,
        );
        stack.push(
            move_op_timed(1, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 200),
            true,
        );
        stack.push(
            move_op_timed(1, [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], 300),
            true,
        );

        // All three should have merged into one operation.
        assert_eq!(stack.undo_count(), 1);

        // The merged operation should have old_pos = (0,0,0), new_pos = (3,0,0).
        let top = stack.undo_stack.last().unwrap();
        let move_op = top.as_any().downcast_ref::<MoveEntityOp>().unwrap();
        assert_eq!(move_op.old_pos, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(move_op.new_pos, Vec3::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn no_merge_different_entities() {
        let mut stack = UndoStack::with_default_size();

        stack.push(
            move_op_timed(1, [0.0; 3], [1.0, 0.0, 0.0], 100),
            true,
        );
        stack.push(
            move_op_timed(2, [0.0; 3], [2.0, 0.0, 0.0], 200),
            true,
        );

        // Different entities should not merge.
        assert_eq!(stack.undo_count(), 2);
    }

    #[test]
    fn no_merge_time_gap() {
        let mut stack = UndoStack::with_default_size();

        stack.push(
            move_op_timed(1, [0.0; 3], [1.0, 0.0, 0.0], 100),
            true,
        );
        stack.push(
            move_op_timed(1, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 1000),
            true,
        );

        // Time gap > 500ms should prevent merge.
        assert_eq!(stack.undo_count(), 2);
    }

    #[test]
    fn no_merge_different_types() {
        let mut stack = UndoStack::with_default_size();

        stack.push(
            move_op_timed(1, [0.0; 3], [1.0, 0.0, 0.0], 100),
            true,
        );
        stack.push(
            Box::new(
                RotateEntityOp::new(EntityId(1), Quat::IDENTITY, Quat::new(0.0, 0.707, 0.0, 0.707))
                    .with_timestamp(200),
            ),
            true,
        );

        // Different operation types should not merge.
        assert_eq!(stack.undo_count(), 2);
    }

    #[test]
    fn merge_disabled() {
        let mut stack = UndoStack::with_default_size();
        stack.set_merge_enabled(false);

        stack.push(
            move_op_timed(1, [0.0; 3], [1.0, 0.0, 0.0], 100),
            true,
        );
        stack.push(
            move_op_timed(1, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 200),
            true,
        );

        // Merge disabled, so both operations should be separate.
        assert_eq!(stack.undo_count(), 2);
    }

    // -- Clear ------------------------------------------------------------------

    #[test]
    fn clear() {
        let mut stack = UndoStack::with_default_size();
        stack.push(move_op(1, [0.0; 3], [1.0, 0.0, 0.0]), false);
        stack.push(move_op(1, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]), false);
        stack.undo();

        assert!(stack.can_undo());
        assert!(stack.can_redo());

        stack.clear();
        assert!(!stack.can_undo());
        assert!(!stack.can_redo());
        assert_eq!(stack.undo_count(), 0);
        assert_eq!(stack.redo_count(), 0);
    }

    // -- Descriptions -----------------------------------------------------------

    #[test]
    fn descriptions() {
        let mut stack = UndoStack::with_default_size();

        stack.push(move_op(1, [0.0; 3], [1.0, 0.0, 0.0]), false);
        assert_eq!(stack.undo_description(), Some("Move Entity"));
        assert_eq!(stack.redo_description(), None);

        stack.undo();
        assert_eq!(stack.undo_description(), None);
        assert_eq!(stack.redo_description(), Some("Move Entity"));
    }

    #[test]
    fn history() {
        let mut stack = UndoStack::with_default_size();

        stack.push(move_op(1, [0.0; 3], [1.0, 0.0, 0.0]), false);
        stack.push(
            Box::new(RotateEntityOp::new(
                EntityId(1),
                Quat::IDENTITY,
                Quat::new(0.0, 0.707, 0.0, 0.707),
            )),
            false,
        );

        let history = stack.undo_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0], "Rotate Entity");
        assert_eq!(history[1], "Move Entity");
    }

    // -- CompoundOp tests -------------------------------------------------------

    #[test]
    fn compound_op() {
        let mut compound = CompoundOp::new("Move and Rotate");
        compound.push(move_op(1, [0.0; 3], [1.0, 0.0, 0.0]));
        compound.push(Box::new(RotateEntityOp::new(
            EntityId(1),
            Quat::IDENTITY,
            Quat::new(0.0, 0.707, 0.0, 0.707),
        )));

        assert_eq!(compound.len(), 2);
        assert_eq!(compound.description(), "Move and Rotate");

        let result = compound.execute();
        assert!(result.is_success());

        let undo_result = compound.undo();
        assert!(undo_result.is_success());
    }

    #[test]
    fn compound_op_builder() {
        let mut stack = UndoStack::with_default_size();
        let mut builder = stack.begin_compound("Transform");
        builder.add(move_op(1, [0.0; 3], [1.0, 0.0, 0.0]));
        builder.add(Box::new(ScaleEntityOp::new(
            EntityId(1),
            Vec3::ONE,
            Vec3::new(2.0, 2.0, 2.0),
        )));

        assert_eq!(builder.len(), 2);
        assert!(!builder.is_empty());

        let compound = builder.finish();
        stack.push(compound, false);

        assert_eq!(stack.undo_count(), 1);

        let desc = stack.undo().unwrap();
        assert_eq!(desc, "Transform");
    }

    // -- undo_n / redo_n --------------------------------------------------------

    #[test]
    fn undo_n() {
        let mut stack = UndoStack::with_default_size();

        for i in 0..5 {
            stack.push(
                move_op(1, [i as f32, 0.0, 0.0], [(i + 1) as f32, 0.0, 0.0]),
                false,
            );
        }

        let descs = stack.undo_n(3);
        assert_eq!(descs.len(), 3);
        assert_eq!(stack.undo_count(), 2);
        assert_eq!(stack.redo_count(), 3);
    }

    #[test]
    fn redo_n() {
        let mut stack = UndoStack::with_default_size();

        for i in 0..5 {
            stack.push(
                move_op(1, [i as f32, 0.0, 0.0], [(i + 1) as f32, 0.0, 0.0]),
                false,
            );
        }

        stack.undo_n(5);
        assert_eq!(stack.redo_count(), 5);

        let descs = stack.redo_n(3);
        assert_eq!(descs.len(), 3);
        assert_eq!(stack.undo_count(), 3);
        assert_eq!(stack.redo_count(), 2);
    }

    // -- Operation types --------------------------------------------------------

    #[test]
    fn move_op_noop_same_position() {
        let op = MoveEntityOp::new(
            EntityId(1),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 2.0, 3.0),
        );
        assert!(matches!(op.execute(), OperationResult::NoOp));
    }

    #[test]
    fn rotate_op() {
        let op = RotateEntityOp::new(
            EntityId(1),
            Quat::IDENTITY,
            Quat::new(0.0, 0.707, 0.0, 0.707),
        );
        assert_eq!(op.description(), "Rotate Entity");
        assert_eq!(op.op_type(), OperationType::Rotate);
        assert!(op.execute().is_success());
        assert!(op.undo().is_success());
    }

    #[test]
    fn scale_op() {
        let op = ScaleEntityOp::new(
            EntityId(1),
            Vec3::ONE,
            Vec3::new(2.0, 2.0, 2.0),
        );
        assert_eq!(op.description(), "Scale Entity");
        assert_eq!(op.op_type(), OperationType::Scale);
    }

    #[test]
    fn spawn_op() {
        let data = SerializedEntityData {
            entity: EntityId(1),
            name: "TestEntity".to_string(),
            components: vec![],
            parent: None,
            children: vec![],
        };
        let op = SpawnEntityOp::new(data);
        assert_eq!(op.description(), "Spawn Entity");
        assert!(op.execute().is_success());
    }

    #[test]
    fn despawn_op() {
        let data = SerializedEntityData {
            entity: EntityId(1),
            name: "TestEntity".to_string(),
            components: vec![SerializedComponentData {
                type_name: "Position".to_string(),
                data: vec![1, 2, 3, 4],
            }],
            parent: None,
            children: vec![],
        };
        let op = DespawnEntityOp::new(data);
        assert_eq!(op.description(), "Delete Entity");
    }

    #[test]
    fn change_component_op() {
        let op = ChangeComponentOp::new(
            EntityId(1),
            "Position",
            "x",
            vec![0, 0, 128, 63], // 1.0f32
            vec![0, 0, 0, 64],   // 2.0f32
        );
        assert_eq!(op.description(), "Change Component");
        assert_eq!(op.op_type(), OperationType::ChangeComponent);
    }

    #[test]
    fn reparent_op() {
        let op = ReparentOp::new(
            EntityId(3),
            Some(EntityId(1)),
            Some(EntityId(2)),
        );
        assert_eq!(op.description(), "Reparent Entity");
        assert_eq!(op.op_type(), OperationType::Reparent);
    }

    #[test]
    fn change_component_merge() {
        let mut stack = UndoStack::with_default_size();

        stack.push(
            Box::new(
                ChangeComponentOp::new(
                    EntityId(1),
                    "Transform",
                    "position.x",
                    vec![0],
                    vec![1],
                )
                .with_timestamp(100),
            ),
            true,
        );

        stack.push(
            Box::new(
                ChangeComponentOp::new(
                    EntityId(1),
                    "Transform",
                    "position.x",
                    vec![1],
                    vec![2],
                )
                .with_timestamp(200),
            ),
            true,
        );

        // Should have merged.
        assert_eq!(stack.undo_count(), 1);
    }

    // -- Vec3 / Quat helper tests -----------------------------------------------

    #[test]
    fn vec3_distance() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn vec3_display() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(format!("{}", v), "(1.00, 2.00, 3.00)");
    }

    #[test]
    fn quat_angle_to() {
        let a = Quat::IDENTITY;
        let b = Quat::IDENTITY;
        assert!(a.angle_to(&b) < 1e-5);
    }

    #[test]
    fn entity_id_display() {
        let id = EntityId(42);
        assert_eq!(format!("{}", id), "Entity(42)");
    }

    // -- UndoStack on_change callback -------------------------------------------

    #[test]
    fn on_change_callback() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut stack = UndoStack::with_default_size();
        stack.set_on_change(move |_event| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        stack.push(move_op(1, [0.0; 3], [1.0, 0.0, 0.0]), false);
        stack.undo();
        stack.redo();

        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    // -- Default stack ----------------------------------------------------------

    #[test]
    fn default_stack() {
        let stack = UndoStack::default();
        assert_eq!(stack.max_size(), 100);
        assert!(!stack.can_undo());
        assert!(!stack.can_redo());
    }

    #[test]
    fn debug_format() {
        let stack = UndoStack::with_default_size();
        let debug = format!("{:?}", stack);
        assert!(debug.contains("UndoStack"));
        assert!(debug.contains("undo_count"));
    }

    // -- OperationResult tests --------------------------------------------------

    #[test]
    fn operation_result_checks() {
        assert!(OperationResult::Success.is_success());
        assert!(!OperationResult::Success.is_failed());
        assert!(OperationResult::Failed("error".to_string()).is_failed());
        assert!(!OperationResult::NoOp.is_success());
        assert!(!OperationResult::NoOp.is_failed());
    }
}
