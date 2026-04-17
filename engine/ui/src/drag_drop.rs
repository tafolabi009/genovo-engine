//! Drag and Drop System
//!
//! Provides a complete drag-and-drop framework for UI elements, including:
//!
//! - `DragSource` and `DropTarget` components
//! - Drag visual (ghost image) management
//! - Drop validation (can this item go here?)
//! - Typed drag payloads
//! - Cross-panel drag operations
//! - Snap-to-slot positioning
//! - List reordering via drag
//!
//! # Architecture
//!
//! ```text
//!  DragSource ──> DragSession ──> DropTarget
//!       │              │              │
//!   DragPayload    DragVisual    DropValidator
//!                      │
//!                 SnapSlotGrid
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut manager = DragDropManager::new();
//! manager.register_source(DragSource::new("inventory_item", payload));
//! manager.register_target(DropTarget::new("equipment_slot", validator));
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// DragId / DropId
// ---------------------------------------------------------------------------

/// Unique identifier for a drag source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DragSourceId {
    pub index: u64,
}

impl DragSourceId {
    pub fn from_raw(index: u64) -> Self {
        Self { index }
    }
}

impl fmt::Display for DragSourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DragSource({})", self.index)
    }
}

/// Unique identifier for a drop target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DropTargetId {
    pub index: u64,
}

impl DropTargetId {
    pub fn from_raw(index: u64) -> Self {
        Self { index }
    }
}

impl fmt::Display for DropTargetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DropTarget({})", self.index)
    }
}

/// Unique identifier for a drag session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DragSessionId {
    pub index: u64,
}

impl DragSessionId {
    pub fn from_raw(index: u64) -> Self {
        Self { index }
    }
}

impl fmt::Display for DragSessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DragSession({})", self.index)
    }
}

static NEXT_DRAG_SOURCE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_DROP_TARGET_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_DRAG_SESSION_ID: AtomicU64 = AtomicU64::new(1);

fn next_drag_source_id() -> DragSourceId {
    DragSourceId {
        index: NEXT_DRAG_SOURCE_ID.fetch_add(1, Ordering::Relaxed),
    }
}

fn next_drop_target_id() -> DropTargetId {
    DropTargetId {
        index: NEXT_DROP_TARGET_ID.fetch_add(1, Ordering::Relaxed),
    }
}

fn next_drag_session_id() -> DragSessionId {
    DragSessionId {
        index: NEXT_DRAG_SESSION_ID.fetch_add(1, Ordering::Relaxed),
    }
}

// ---------------------------------------------------------------------------
// DragPayloadType
// ---------------------------------------------------------------------------

/// The type of data being dragged. Used for drop validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DragPayloadType {
    /// An inventory item (e.g., weapon, potion).
    InventoryItem,
    /// An asset from the asset browser (editor).
    Asset,
    /// A UI widget being rearranged.
    Widget,
    /// A list item being reordered.
    ListItem,
    /// A tree node being moved.
    TreeNode,
    /// A color swatch or color value.
    Color,
    /// A file or folder reference.
    File,
    /// A custom payload type identified by a string.
    Custom(String),
}

impl fmt::Display for DragPayloadType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InventoryItem => write!(f, "InventoryItem"),
            Self::Asset => write!(f, "Asset"),
            Self::Widget => write!(f, "Widget"),
            Self::ListItem => write!(f, "ListItem"),
            Self::TreeNode => write!(f, "TreeNode"),
            Self::Color => write!(f, "Color"),
            Self::File => write!(f, "File"),
            Self::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

// ---------------------------------------------------------------------------
// DragPayload
// ---------------------------------------------------------------------------

/// Data being carried during a drag operation.
///
/// The payload contains the type of data and arbitrary key-value metadata
/// that the drop target can inspect to decide whether to accept the drop.
#[derive(Debug, Clone)]
pub struct DragPayload {
    /// The type of this payload.
    pub payload_type: DragPayloadType,
    /// A unique identifier for the dragged item.
    pub item_id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Optional icon/image asset identifier for the drag visual.
    pub icon_asset: Option<String>,
    /// Arbitrary metadata key-value pairs.
    pub metadata: HashMap<String, String>,
    /// The source panel or container identifier.
    pub source_panel: Option<String>,
    /// The original index in a list (for reordering).
    pub source_index: Option<usize>,
}

impl DragPayload {
    /// Creates a new drag payload.
    pub fn new(
        payload_type: DragPayloadType,
        item_id: impl Into<String>,
        display_name: impl Into<String>,
    ) -> Self {
        Self {
            payload_type,
            item_id: item_id.into(),
            display_name: display_name.into(),
            icon_asset: None,
            metadata: HashMap::new(),
            source_panel: None,
            source_index: None,
        }
    }

    /// Sets the icon asset.
    pub fn with_icon(mut self, asset: impl Into<String>) -> Self {
        self.icon_asset = Some(asset.into());
        self
    }

    /// Adds a metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Sets the source panel.
    pub fn with_source_panel(mut self, panel: impl Into<String>) -> Self {
        self.source_panel = Some(panel.into());
        self
    }

    /// Sets the source index (for list reordering).
    pub fn with_source_index(mut self, index: usize) -> Self {
        self.source_index = Some(index);
        self
    }

    /// Returns a metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// DragVisual
// ---------------------------------------------------------------------------

/// Configuration for the visual representation shown while dragging.
#[derive(Debug, Clone)]
pub struct DragVisual {
    /// The type of visual to display.
    pub visual_type: DragVisualType,
    /// Offset from the cursor position (in logical pixels).
    pub cursor_offset_x: f32,
    /// Offset from the cursor position (in logical pixels).
    pub cursor_offset_y: f32,
    /// Width of the drag visual.
    pub width: f32,
    /// Height of the drag visual.
    pub height: f32,
    /// Opacity of the drag visual (0.0 to 1.0).
    pub opacity: f32,
    /// Scale factor applied to the visual.
    pub scale: f32,
    /// Whether to show a count badge (for multi-select drags).
    pub show_count: bool,
    /// The count to display on the badge.
    pub count: usize,
    /// Rotation in degrees.
    pub rotation: f32,
}

/// The type of visual representation during a drag.
#[derive(Debug, Clone)]
pub enum DragVisualType {
    /// A ghost image of the original item.
    GhostImage {
        /// Asset ID for the ghost image.
        asset_id: String,
    },
    /// A custom-rendered widget.
    CustomWidget {
        /// Identifier for the widget template.
        widget_template_id: String,
    },
    /// Just show the item's icon.
    IconOnly {
        /// Asset ID for the icon.
        icon_asset: String,
    },
    /// A simple rectangle with the item's name.
    TextLabel {
        /// The text to display.
        text: String,
    },
    /// No visual (the cursor change indicates dragging).
    None,
}

impl DragVisual {
    /// Creates a ghost image visual.
    pub fn ghost_image(asset_id: impl Into<String>, width: f32, height: f32) -> Self {
        Self {
            visual_type: DragVisualType::GhostImage {
                asset_id: asset_id.into(),
            },
            cursor_offset_x: -width / 2.0,
            cursor_offset_y: -height / 2.0,
            width,
            height,
            opacity: 0.7,
            scale: 1.0,
            show_count: false,
            count: 0,
            rotation: 0.0,
        }
    }

    /// Creates an icon-only visual.
    pub fn icon(icon_asset: impl Into<String>, size: f32) -> Self {
        Self {
            visual_type: DragVisualType::IconOnly {
                icon_asset: icon_asset.into(),
            },
            cursor_offset_x: -size / 2.0,
            cursor_offset_y: -size / 2.0,
            width: size,
            height: size,
            opacity: 0.8,
            scale: 1.0,
            show_count: false,
            count: 0,
            rotation: 0.0,
        }
    }

    /// Creates a text label visual.
    pub fn text_label(text: impl Into<String>) -> Self {
        Self {
            visual_type: DragVisualType::TextLabel {
                text: text.into(),
            },
            cursor_offset_x: 10.0,
            cursor_offset_y: 10.0,
            width: 150.0,
            height: 30.0,
            opacity: 0.9,
            scale: 1.0,
            show_count: false,
            count: 0,
            rotation: 0.0,
        }
    }

    /// Creates a visual with no graphical representation.
    pub fn none() -> Self {
        Self {
            visual_type: DragVisualType::None,
            cursor_offset_x: 0.0,
            cursor_offset_y: 0.0,
            width: 0.0,
            height: 0.0,
            opacity: 0.0,
            scale: 1.0,
            show_count: false,
            count: 0,
            rotation: 0.0,
        }
    }

    /// Sets the cursor offset.
    pub fn with_offset(mut self, x: f32, y: f32) -> Self {
        self.cursor_offset_x = x;
        self.cursor_offset_y = y;
        self
    }

    /// Sets the opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Sets the scale factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Enables the count badge.
    pub fn with_count(mut self, count: usize) -> Self {
        self.show_count = true;
        self.count = count;
        self
    }

    /// Sets the rotation in degrees.
    pub fn with_rotation(mut self, degrees: f32) -> Self {
        self.rotation = degrees;
        self
    }

    /// Calculates the screen position of the visual given cursor position.
    pub fn screen_position(&self, cursor_x: f32, cursor_y: f32) -> (f32, f32) {
        (
            cursor_x + self.cursor_offset_x,
            cursor_y + self.cursor_offset_y,
        )
    }

    /// Returns the bounding rectangle of the visual at the given cursor position.
    pub fn bounds_at_cursor(&self, cursor_x: f32, cursor_y: f32) -> DragRect {
        let (x, y) = self.screen_position(cursor_x, cursor_y);
        let w = self.width * self.scale;
        let h = self.height * self.scale;
        DragRect {
            x,
            y,
            width: w,
            height: h,
        }
    }
}

// ---------------------------------------------------------------------------
// DragRect
// ---------------------------------------------------------------------------

/// A simple axis-aligned rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DragRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl DragRect {
    /// Creates a new rectangle.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Returns `true` if the point is inside this rectangle.
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width && py >= self.y && py <= self.y + self.height
    }

    /// Returns `true` if this rectangle overlaps with another.
    pub fn overlaps(&self, other: &DragRect) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }

    /// Returns the center point.
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Returns the distance from the center to a point.
    pub fn distance_to(&self, px: f32, py: f32) -> f32 {
        let (cx, cy) = self.center();
        let dx = px - cx;
        let dy = py - cy;
        (dx * dx + dy * dy).sqrt()
    }
}

// ---------------------------------------------------------------------------
// DropValidation
// ---------------------------------------------------------------------------

/// Result of validating whether a drop is accepted at a target.
#[derive(Debug, Clone)]
pub enum DropValidation {
    /// The drop is accepted. Show positive feedback.
    Accepted,
    /// The drop is rejected. Show negative feedback.
    Rejected {
        /// Reason the drop was rejected.
        reason: String,
    },
    /// The drop can be accepted with a warning/condition.
    Conditional {
        /// Warning or condition message.
        message: String,
    },
}

impl DropValidation {
    /// Returns `true` if the drop would be accepted.
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted | Self::Conditional { .. })
    }

    /// Returns `true` if the drop would be rejected.
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected { .. })
    }
}

// ---------------------------------------------------------------------------
// DropValidator
// ---------------------------------------------------------------------------

/// Validates whether a drag payload can be dropped on a target.
pub struct DropValidator {
    /// Human-readable name for debugging.
    pub name: String,
    /// The accepted payload types.
    pub accepted_types: Vec<DragPayloadType>,
    /// Custom validation function.
    validate_fn: Option<Box<dyn Fn(&DragPayload) -> DropValidation + Send + Sync>>,
}

impl DropValidator {
    /// Creates a validator that accepts specific payload types.
    pub fn accept_types(name: impl Into<String>, types: Vec<DragPayloadType>) -> Self {
        Self {
            name: name.into(),
            accepted_types: types,
            validate_fn: None,
        }
    }

    /// Creates a validator with a custom validation function.
    pub fn custom<F>(name: impl Into<String>, validate_fn: F) -> Self
    where
        F: Fn(&DragPayload) -> DropValidation + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            accepted_types: Vec::new(),
            validate_fn: Some(Box::new(validate_fn)),
        }
    }

    /// Creates a validator that accepts all payload types.
    pub fn accept_all(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            accepted_types: Vec::new(),
            validate_fn: Some(Box::new(|_| DropValidation::Accepted)),
        }
    }

    /// Validates a drag payload against this validator.
    pub fn validate(&self, payload: &DragPayload) -> DropValidation {
        // Check custom validator first.
        if let Some(ref validate_fn) = self.validate_fn {
            return validate_fn(payload);
        }

        // Check accepted types.
        if self.accepted_types.is_empty() {
            return DropValidation::Accepted;
        }

        if self.accepted_types.contains(&payload.payload_type) {
            DropValidation::Accepted
        } else {
            DropValidation::Rejected {
                reason: format!(
                    "Payload type '{}' not accepted by '{}'",
                    payload.payload_type, self.name
                ),
            }
        }
    }
}

impl fmt::Debug for DropValidator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DropValidator")
            .field("name", &self.name)
            .field("accepted_types", &self.accepted_types)
            .field("has_custom_fn", &self.validate_fn.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DragSource
// ---------------------------------------------------------------------------

/// A UI element that can be dragged.
#[derive(Debug)]
pub struct DragSource {
    /// Unique identifier.
    pub id: DragSourceId,
    /// Human-readable name.
    pub name: String,
    /// The screen bounds of this drag source.
    pub bounds: DragRect,
    /// Whether this source is currently enabled for dragging.
    pub enabled: bool,
    /// The payload to carry when dragged.
    pub payload: DragPayload,
    /// The visual to show during dragging.
    pub drag_visual: DragVisual,
    /// Whether the source item should be hidden while being dragged.
    pub hide_on_drag: bool,
    /// Minimum distance (in pixels) the cursor must move before drag starts.
    pub drag_threshold: f32,
    /// Optional group identifier for multi-select dragging.
    pub group: Option<String>,
    /// Whether this source allows being dragged to a different panel.
    pub allow_cross_panel: bool,
}

impl DragSource {
    /// Creates a new drag source.
    pub fn new(
        name: impl Into<String>,
        bounds: DragRect,
        payload: DragPayload,
    ) -> Self {
        let name_str = name.into();
        let visual = DragVisual::text_label(&name_str);
        Self {
            id: next_drag_source_id(),
            name: name_str,
            bounds,
            enabled: true,
            payload,
            drag_visual: visual,
            hide_on_drag: false,
            drag_threshold: 5.0,
            group: None,
            allow_cross_panel: true,
        }
    }

    /// Sets the drag visual.
    pub fn with_visual(mut self, visual: DragVisual) -> Self {
        self.drag_visual = visual;
        self
    }

    /// Sets whether to hide the source while dragging.
    pub fn with_hide_on_drag(mut self, hide: bool) -> Self {
        self.hide_on_drag = hide;
        self
    }

    /// Sets the drag threshold distance.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.drag_threshold = threshold;
        self
    }

    /// Sets the group for multi-select.
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    /// Returns `true` if the given point is within this source's bounds.
    pub fn hit_test(&self, x: f32, y: f32) -> bool {
        self.enabled && self.bounds.contains(x, y)
    }
}

// ---------------------------------------------------------------------------
// DropTarget
// ---------------------------------------------------------------------------

/// A UI element that can receive dropped items.
#[derive(Debug)]
pub struct DropTarget {
    /// Unique identifier.
    pub id: DropTargetId,
    /// Human-readable name.
    pub name: String,
    /// The screen bounds of this drop target.
    pub bounds: DragRect,
    /// Whether this target is currently enabled.
    pub enabled: bool,
    /// The validator for incoming drops.
    pub validator: DropValidator,
    /// Visual feedback configuration.
    pub feedback: DropFeedback,
    /// Optional panel identifier for cross-panel drops.
    pub panel: Option<String>,
    /// Whether this target supports reordering (list mode).
    pub supports_reorder: bool,
    /// Snap slots within this target.
    pub snap_slots: Vec<SnapSlot>,
    /// Priority for overlapping targets (higher = preferred).
    pub priority: i32,
}

/// Visual feedback configuration for drop targets.
#[derive(Debug, Clone)]
pub struct DropFeedback {
    /// Whether to highlight the target when a compatible item hovers over it.
    pub highlight_on_hover: bool,
    /// Color tint to apply when hovering with a compatible payload.
    pub accept_tint: [f32; 4],
    /// Color tint to apply when hovering with an incompatible payload.
    pub reject_tint: [f32; 4],
    /// Whether to show a drop indicator line (for list reordering).
    pub show_insert_indicator: bool,
    /// Width of the insert indicator line.
    pub indicator_width: f32,
}

impl Default for DropFeedback {
    fn default() -> Self {
        Self {
            highlight_on_hover: true,
            accept_tint: [0.0, 1.0, 0.0, 0.3],
            reject_tint: [1.0, 0.0, 0.0, 0.3],
            show_insert_indicator: false,
            indicator_width: 2.0,
        }
    }
}

impl DropTarget {
    /// Creates a new drop target.
    pub fn new(
        name: impl Into<String>,
        bounds: DragRect,
        validator: DropValidator,
    ) -> Self {
        Self {
            id: next_drop_target_id(),
            name: name.into(),
            bounds,
            enabled: true,
            validator,
            feedback: DropFeedback::default(),
            panel: None,
            supports_reorder: false,
            snap_slots: Vec::new(),
            priority: 0,
        }
    }

    /// Sets the panel identifier.
    pub fn with_panel(mut self, panel: impl Into<String>) -> Self {
        self.panel = Some(panel.into());
        self
    }

    /// Enables list reordering mode.
    pub fn with_reorder(mut self) -> Self {
        self.supports_reorder = true;
        self.feedback.show_insert_indicator = true;
        self
    }

    /// Adds a snap slot.
    pub fn with_snap_slot(mut self, slot: SnapSlot) -> Self {
        self.snap_slots.push(slot);
        self
    }

    /// Sets the priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Returns `true` if the given point is within this target's bounds.
    pub fn hit_test(&self, x: f32, y: f32) -> bool {
        self.enabled && self.bounds.contains(x, y)
    }

    /// Validates a payload against this target.
    pub fn validate(&self, payload: &DragPayload) -> DropValidation {
        if !self.enabled {
            return DropValidation::Rejected {
                reason: "Target is disabled".to_string(),
            };
        }
        self.validator.validate(payload)
    }

    /// Finds the nearest snap slot to the given position.
    pub fn nearest_snap_slot(&self, x: f32, y: f32) -> Option<(usize, &SnapSlot)> {
        if self.snap_slots.is_empty() {
            return None;
        }

        let mut best_index = 0;
        let mut best_distance = f32::MAX;

        for (i, slot) in self.snap_slots.iter().enumerate() {
            if !slot.available {
                continue;
            }
            let dx = x - slot.center_x;
            let dy = y - slot.center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < best_distance && dist <= slot.snap_radius {
                best_distance = dist;
                best_index = i;
            }
        }

        if best_distance <= self.snap_slots[best_index].snap_radius {
            Some((best_index, &self.snap_slots[best_index]))
        } else {
            None
        }
    }

    /// For reorderable lists: determines the insertion index based on cursor
    /// position relative to existing items.
    pub fn reorder_index(&self, cursor_y: f32, item_count: usize, item_height: f32) -> usize {
        if item_count == 0 {
            return 0;
        }
        let relative_y = cursor_y - self.bounds.y;
        let index = (relative_y / item_height).floor() as usize;
        index.min(item_count)
    }
}

// ---------------------------------------------------------------------------
// SnapSlot
// ---------------------------------------------------------------------------

/// A specific position within a drop target where items can snap to.
#[derive(Debug, Clone)]
pub struct SnapSlot {
    /// Slot index/identifier.
    pub index: usize,
    /// Human-readable label.
    pub label: String,
    /// Center X position in screen coordinates.
    pub center_x: f32,
    /// Center Y position in screen coordinates.
    pub center_y: f32,
    /// Width of the slot.
    pub width: f32,
    /// Height of the slot.
    pub height: f32,
    /// Maximum distance for snapping.
    pub snap_radius: f32,
    /// Whether this slot is currently available (not occupied).
    pub available: bool,
    /// Optional accepted payload types for this specific slot.
    pub accepted_types: Vec<DragPayloadType>,
    /// Optional data associated with the currently occupying item.
    pub occupant_id: Option<String>,
}

impl SnapSlot {
    /// Creates a new snap slot at the given position.
    pub fn new(
        index: usize,
        label: impl Into<String>,
        center_x: f32,
        center_y: f32,
        width: f32,
        height: f32,
    ) -> Self {
        Self {
            index,
            label: label.into(),
            center_x,
            center_y,
            width,
            height,
            snap_radius: (width.max(height)) * 0.75,
            available: true,
            accepted_types: Vec::new(),
            occupant_id: None,
        }
    }

    /// Sets the snap radius.
    pub fn with_snap_radius(mut self, radius: f32) -> Self {
        self.snap_radius = radius;
        self
    }

    /// Sets accepted types for this slot.
    pub fn with_accepted_types(mut self, types: Vec<DragPayloadType>) -> Self {
        self.accepted_types = types;
        self
    }

    /// Returns `true` if this slot accepts the given payload type.
    pub fn accepts(&self, payload_type: &DragPayloadType) -> bool {
        if self.accepted_types.is_empty() {
            return true;
        }
        self.accepted_types.contains(payload_type)
    }

    /// Occupies this slot with the given item.
    pub fn occupy(&mut self, item_id: impl Into<String>) {
        self.occupant_id = Some(item_id.into());
        self.available = false;
    }

    /// Vacates this slot.
    pub fn vacate(&mut self) {
        self.occupant_id = None;
        self.available = true;
    }

    /// Returns the bounding rectangle of this slot.
    pub fn bounds(&self) -> DragRect {
        DragRect {
            x: self.center_x - self.width / 2.0,
            y: self.center_y - self.height / 2.0,
            width: self.width,
            height: self.height,
        }
    }
}

// ---------------------------------------------------------------------------
// DragSession
// ---------------------------------------------------------------------------

/// Represents an active drag operation.
#[derive(Debug)]
pub struct DragSession {
    /// Unique session identifier.
    pub id: DragSessionId,
    /// The drag source that initiated this session.
    pub source_id: DragSourceId,
    /// The payload being dragged.
    pub payload: DragPayload,
    /// The drag visual configuration.
    pub visual: DragVisual,
    /// The current phase of the drag operation.
    pub phase: DragPhase,
    /// Screen position where the drag started.
    pub start_x: f32,
    /// Screen position where the drag started.
    pub start_y: f32,
    /// Current cursor X position.
    pub current_x: f32,
    /// Current cursor Y position.
    pub current_y: f32,
    /// The drop target currently being hovered over.
    pub hover_target: Option<DropTargetId>,
    /// The validation result for the current hover target.
    pub hover_validation: Option<DropValidation>,
    /// The snap slot currently being hovered over.
    pub hover_snap_slot: Option<usize>,
    /// Time elapsed since drag started (in seconds).
    pub elapsed: f32,
    /// Whether this drag has been cancelled.
    pub cancelled: bool,
}

/// The current phase of a drag operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DragPhase {
    /// The user pressed down but hasn't exceeded the drag threshold.
    Pending,
    /// The drag is actively in progress.
    Dragging,
    /// The drag has been dropped.
    Dropped,
    /// The drag has been cancelled (e.g., by pressing Escape).
    Cancelled,
}

impl DragSession {
    /// Creates a new drag session.
    pub fn new(
        source_id: DragSourceId,
        payload: DragPayload,
        visual: DragVisual,
        start_x: f32,
        start_y: f32,
    ) -> Self {
        Self {
            id: next_drag_session_id(),
            source_id,
            payload,
            visual,
            phase: DragPhase::Pending,
            start_x,
            start_y,
            current_x: start_x,
            current_y: start_y,
            hover_target: None,
            hover_validation: None,
            hover_snap_slot: None,
            elapsed: 0.0,
            cancelled: false,
        }
    }

    /// Returns the distance the cursor has moved from the start position.
    pub fn drag_distance(&self) -> f32 {
        let dx = self.current_x - self.start_x;
        let dy = self.current_y - self.start_y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Updates the cursor position.
    pub fn update_position(&mut self, x: f32, y: f32) {
        self.current_x = x;
        self.current_y = y;
    }

    /// Updates the elapsed time.
    pub fn update_time(&mut self, dt: f32) {
        self.elapsed += dt;
    }

    /// Returns `true` if the drag is actively in progress.
    pub fn is_active(&self) -> bool {
        self.phase == DragPhase::Dragging
    }

    /// Returns `true` if the drag has completed (dropped or cancelled).
    pub fn is_complete(&self) -> bool {
        matches!(self.phase, DragPhase::Dropped | DragPhase::Cancelled)
    }
}

// ---------------------------------------------------------------------------
// DragDropEvent
// ---------------------------------------------------------------------------

/// Events emitted by the drag-and-drop system.
#[derive(Debug, Clone)]
pub enum DragDropEvent {
    /// A drag operation has started.
    DragStarted {
        session_id: DragSessionId,
        source_id: DragSourceId,
        payload_type: DragPayloadType,
    },
    /// The cursor is hovering over a valid drop target.
    DragEnterTarget {
        session_id: DragSessionId,
        target_id: DropTargetId,
        validation: DropValidation,
    },
    /// The cursor has left a drop target.
    DragLeaveTarget {
        session_id: DragSessionId,
        target_id: DropTargetId,
    },
    /// A drop was successfully completed.
    DropCompleted {
        session_id: DragSessionId,
        source_id: DragSourceId,
        target_id: DropTargetId,
        payload: DragPayload,
        snap_slot: Option<usize>,
    },
    /// A drop was rejected by the target.
    DropRejected {
        session_id: DragSessionId,
        target_id: DropTargetId,
        reason: String,
    },
    /// The drag was cancelled.
    DragCancelled {
        session_id: DragSessionId,
        source_id: DragSourceId,
    },
    /// A list reorder occurred.
    ListReordered {
        session_id: DragSessionId,
        target_id: DropTargetId,
        from_index: usize,
        to_index: usize,
    },
}

// ---------------------------------------------------------------------------
// DragDropManager
// ---------------------------------------------------------------------------

/// Central manager for all drag-and-drop operations.
///
/// The manager owns drag sources, drop targets, and active drag sessions.
/// It processes cursor input and generates events for the UI system to
/// handle.
pub struct DragDropManager {
    /// All registered drag sources.
    sources: HashMap<DragSourceId, DragSource>,
    /// All registered drop targets.
    targets: HashMap<DropTargetId, DropTarget>,
    /// The currently active drag session, if any.
    active_session: Option<DragSession>,
    /// Pending events to be consumed by the UI system.
    pending_events: Vec<DragDropEvent>,
    /// History of completed drag operations (for undo support).
    history: Vec<DragDropEvent>,
    /// Maximum history size.
    max_history: usize,
    /// Whether to show debug overlays for drag sources and targets.
    debug_draw: bool,
}

impl DragDropManager {
    /// Creates a new drag-and-drop manager.
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            targets: HashMap::new(),
            active_session: None,
            pending_events: Vec::new(),
            history: Vec::new(),
            max_history: 100,
            debug_draw: false,
        }
    }

    /// Registers a drag source.
    pub fn register_source(&mut self, source: DragSource) -> DragSourceId {
        let id = source.id;
        self.sources.insert(id, source);
        id
    }

    /// Unregisters a drag source.
    pub fn unregister_source(&mut self, id: DragSourceId) -> Option<DragSource> {
        self.sources.remove(&id)
    }

    /// Registers a drop target.
    pub fn register_target(&mut self, target: DropTarget) -> DropTargetId {
        let id = target.id;
        self.targets.insert(id, target);
        id
    }

    /// Unregisters a drop target.
    pub fn unregister_target(&mut self, id: DropTargetId) -> Option<DropTarget> {
        self.targets.remove(&id)
    }

    /// Returns a reference to a drag source.
    pub fn get_source(&self, id: DragSourceId) -> Option<&DragSource> {
        self.sources.get(&id)
    }

    /// Returns a mutable reference to a drag source.
    pub fn get_source_mut(&mut self, id: DragSourceId) -> Option<&mut DragSource> {
        self.sources.get_mut(&id)
    }

    /// Returns a reference to a drop target.
    pub fn get_target(&self, id: DropTargetId) -> Option<&DropTarget> {
        self.targets.get(&id)
    }

    /// Returns a mutable reference to a drop target.
    pub fn get_target_mut(&mut self, id: DropTargetId) -> Option<&mut DropTarget> {
        self.targets.get_mut(&id)
    }

    /// Returns `true` if a drag operation is currently in progress.
    pub fn is_dragging(&self) -> bool {
        self.active_session
            .as_ref()
            .map_or(false, |s| s.is_active())
    }

    /// Returns a reference to the active drag session.
    pub fn active_session(&self) -> Option<&DragSession> {
        self.active_session.as_ref()
    }

    /// Initiates a drag operation from the given source at the cursor position.
    pub fn begin_drag(&mut self, source_id: DragSourceId, cursor_x: f32, cursor_y: f32) -> bool {
        if self.is_dragging() {
            return false;
        }

        let source = match self.sources.get(&source_id) {
            Some(s) if s.enabled => s,
            _ => return false,
        };

        let session = DragSession::new(
            source_id,
            source.payload.clone(),
            source.drag_visual.clone(),
            cursor_x,
            cursor_y,
        );

        let session_id = session.id;
        let payload_type = session.payload.payload_type.clone();
        self.active_session = Some(session);

        self.pending_events.push(DragDropEvent::DragStarted {
            session_id,
            source_id,
            payload_type,
        });

        true
    }

    /// Updates the drag cursor position and processes hover targets.
    pub fn update_drag(&mut self, cursor_x: f32, cursor_y: f32) {
        let session = match self.active_session.as_mut() {
            Some(s) => s,
            None => return,
        };

        session.update_position(cursor_x, cursor_y);

        // Check if we've exceeded the drag threshold.
        if session.phase == DragPhase::Pending {
            let source = self.sources.get(&session.source_id);
            let threshold = source.map_or(5.0, |s| s.drag_threshold);
            if session.drag_distance() >= threshold {
                session.phase = DragPhase::Dragging;
            } else {
                return;
            }
        }

        // Find the drop target under the cursor.
        let mut best_target: Option<(DropTargetId, i32)> = None;
        for (id, target) in &self.targets {
            if target.hit_test(cursor_x, cursor_y) {
                let priority = target.priority;
                if best_target.map_or(true, |(_, p)| priority > p) {
                    best_target = Some((*id, priority));
                }
            }
        }

        let new_hover_target = best_target.map(|(id, _)| id);
        let old_hover_target = session.hover_target;

        // Handle target transitions.
        if old_hover_target != new_hover_target {
            let session_id = session.id;

            if let Some(old_id) = old_hover_target {
                self.pending_events.push(DragDropEvent::DragLeaveTarget {
                    session_id,
                    target_id: old_id,
                });
            }

            if let Some(new_id) = new_hover_target {
                let validation = self
                    .targets
                    .get(&new_id)
                    .map(|t| t.validate(&session.payload))
                    .unwrap_or(DropValidation::Rejected {
                        reason: "Target not found".to_string(),
                    });

                self.pending_events.push(DragDropEvent::DragEnterTarget {
                    session_id,
                    target_id: new_id,
                    validation: validation.clone(),
                });

                // Update session state (re-borrow after events).
                if let Some(s) = self.active_session.as_mut() {
                    s.hover_target = new_hover_target;
                    s.hover_validation = Some(validation);
                }
            } else {
                if let Some(s) = self.active_session.as_mut() {
                    s.hover_target = None;
                    s.hover_validation = None;
                }
            }

            // Check snap slots.
            if let Some(target_id) = new_hover_target {
                if let Some(target) = self.targets.get(&target_id) {
                    let snap = target.nearest_snap_slot(cursor_x, cursor_y);
                    if let Some(s) = self.active_session.as_mut() {
                        s.hover_snap_slot = snap.map(|(i, _)| i);
                    }
                }
            }
        } else if let Some(target_id) = new_hover_target {
            // Still hovering the same target: update snap slot.
            if let Some(target) = self.targets.get(&target_id) {
                let snap = target.nearest_snap_slot(cursor_x, cursor_y);
                if let Some(s) = self.active_session.as_mut() {
                    s.hover_snap_slot = snap.map(|(i, _)| i);
                }
            }
        }
    }

    /// Completes the drag operation by dropping at the current position.
    pub fn end_drag(&mut self) -> Option<DragDropEvent> {
        let session = match self.active_session.take() {
            Some(s) => s,
            None => return None,
        };

        if session.phase == DragPhase::Pending {
            // Never exceeded threshold, no drag occurred.
            return None;
        }

        let session_id = session.id;
        let source_id = session.source_id;

        if let Some(target_id) = session.hover_target {
            let validation = self
                .targets
                .get(&target_id)
                .map(|t| t.validate(&session.payload))
                .unwrap_or(DropValidation::Rejected {
                    reason: "Target not found".to_string(),
                });

            match validation {
                DropValidation::Accepted | DropValidation::Conditional { .. } => {
                    // Check for list reorder.
                    if let Some(target) = self.targets.get(&target_id) {
                        if target.supports_reorder {
                            if let Some(from_index) = session.payload.source_index {
                                let to_index = target.reorder_index(
                                    session.current_y,
                                    10, // placeholder for item_count
                                    30.0, // placeholder for item_height
                                );
                                let event = DragDropEvent::ListReordered {
                                    session_id,
                                    target_id,
                                    from_index,
                                    to_index,
                                };
                                self.record_event(event.clone());
                                return Some(event);
                            }
                        }
                    }

                    let event = DragDropEvent::DropCompleted {
                        session_id,
                        source_id,
                        target_id,
                        payload: session.payload,
                        snap_slot: session.hover_snap_slot,
                    };
                    self.record_event(event.clone());
                    Some(event)
                }
                DropValidation::Rejected { reason } => {
                    let event = DragDropEvent::DropRejected {
                        session_id,
                        target_id,
                        reason,
                    };
                    self.record_event(event.clone());
                    Some(event)
                }
            }
        } else {
            let event = DragDropEvent::DragCancelled {
                session_id,
                source_id,
            };
            self.record_event(event.clone());
            Some(event)
        }
    }

    /// Cancels the active drag operation.
    pub fn cancel_drag(&mut self) -> Option<DragDropEvent> {
        let session = match self.active_session.take() {
            Some(s) => s,
            None => return None,
        };

        let event = DragDropEvent::DragCancelled {
            session_id: session.id,
            source_id: session.source_id,
        };
        self.record_event(event.clone());
        Some(event)
    }

    /// Takes and returns all pending events.
    pub fn drain_events(&mut self) -> Vec<DragDropEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Returns the drag operation history.
    pub fn history(&self) -> &[DragDropEvent] {
        &self.history
    }

    /// Clears the history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Returns the number of registered drag sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Returns the number of registered drop targets.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Enables or disables debug drawing.
    pub fn set_debug_draw(&mut self, enabled: bool) {
        self.debug_draw = enabled;
    }

    /// Returns `true` if debug drawing is enabled.
    pub fn debug_draw_enabled(&self) -> bool {
        self.debug_draw
    }

    /// Returns debug information about all sources and targets.
    pub fn debug_info(&self) -> DragDropDebugInfo {
        DragDropDebugInfo {
            source_count: self.sources.len(),
            target_count: self.targets.len(),
            is_dragging: self.is_dragging(),
            active_session_id: self.active_session.as_ref().map(|s| s.id),
            hover_target: self
                .active_session
                .as_ref()
                .and_then(|s| s.hover_target),
            pending_event_count: self.pending_events.len(),
            history_count: self.history.len(),
        }
    }

    /// Records an event to the history.
    fn record_event(&mut self, event: DragDropEvent) {
        self.pending_events.push(event.clone());
        self.history.push(event);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }
}

impl Default for DragDropManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for DragDropManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DragDropManager")
            .field("source_count", &self.sources.len())
            .field("target_count", &self.targets.len())
            .field("is_dragging", &self.is_dragging())
            .field("pending_events", &self.pending_events.len())
            .finish()
    }
}

/// Debug information about the drag-and-drop system.
#[derive(Debug, Clone)]
pub struct DragDropDebugInfo {
    pub source_count: usize,
    pub target_count: usize,
    pub is_dragging: bool,
    pub active_session_id: Option<DragSessionId>,
    pub hover_target: Option<DropTargetId>,
    pub pending_event_count: usize,
    pub history_count: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_payload() -> DragPayload {
        DragPayload::new(DragPayloadType::InventoryItem, "sword_01", "Iron Sword")
    }

    fn make_rect(x: f32, y: f32, w: f32, h: f32) -> DragRect {
        DragRect::new(x, y, w, h)
    }

    #[test]
    fn test_rect_contains() {
        let r = make_rect(10.0, 10.0, 100.0, 100.0);
        assert!(r.contains(50.0, 50.0));
        assert!(!r.contains(5.0, 5.0));
    }

    #[test]
    fn test_rect_overlaps() {
        let a = make_rect(0.0, 0.0, 50.0, 50.0);
        let b = make_rect(25.0, 25.0, 50.0, 50.0);
        assert!(a.overlaps(&b));
        let c = make_rect(100.0, 100.0, 50.0, 50.0);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_drag_source_hit_test() {
        let source = DragSource::new("test", make_rect(0.0, 0.0, 50.0, 50.0), make_payload());
        assert!(source.hit_test(25.0, 25.0));
        assert!(!source.hit_test(100.0, 100.0));
    }

    #[test]
    fn test_drop_validator_accept_types() {
        let validator = DropValidator::accept_types(
            "equipment",
            vec![DragPayloadType::InventoryItem],
        );
        let payload = make_payload();
        assert!(validator.validate(&payload).is_accepted());

        let file_payload = DragPayload::new(DragPayloadType::File, "file.txt", "file.txt");
        assert!(validator.validate(&file_payload).is_rejected());
    }

    #[test]
    fn test_snap_slot() {
        let mut slot = SnapSlot::new(0, "slot_0", 50.0, 50.0, 40.0, 40.0);
        assert!(slot.available);
        slot.occupy("item_1");
        assert!(!slot.available);
        slot.vacate();
        assert!(slot.available);
    }

    #[test]
    fn test_drag_drop_manager_begin() {
        let mut manager = DragDropManager::new();
        let source = DragSource::new("test", make_rect(0.0, 0.0, 50.0, 50.0), make_payload());
        let source_id = manager.register_source(source);
        assert!(manager.begin_drag(source_id, 25.0, 25.0));
        assert!(manager.active_session().is_some());
    }

    #[test]
    fn test_drag_drop_manager_cancel() {
        let mut manager = DragDropManager::new();
        let source = DragSource::new("test", make_rect(0.0, 0.0, 50.0, 50.0), make_payload());
        let source_id = manager.register_source(source);
        manager.begin_drag(source_id, 25.0, 25.0);
        let event = manager.cancel_drag();
        assert!(event.is_some());
        assert!(manager.active_session().is_none());
    }

    #[test]
    fn test_drag_visual_position() {
        let visual = DragVisual::ghost_image("icon.png", 64.0, 64.0);
        let (x, y) = visual.screen_position(100.0, 100.0);
        assert!((x - 68.0).abs() < 0.01);
        assert!((y - 68.0).abs() < 0.01);
    }
}
