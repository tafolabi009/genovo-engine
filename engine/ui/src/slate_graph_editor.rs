//! Enhanced graph editor widget: zoom levels, minimap, node categories,
//! connection validation, auto-layout (force-directed), comment boxes, and
//! reroute nodes.
//!
//! This module provides a full-featured node graph editor suitable for
//! material editors, visual scripting, AI behavior trees, and any other
//! directed graph editing workflow.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// 2D position in graph space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphPos {
    pub x: f32,
    pub y: f32,
}

impl GraphPos {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    pub fn scale(&self, factor: f32) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
        }
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            return *self;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
        }
    }
}

/// 2D size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphSize {
    pub width: f32,
    pub height: f32,
}

impl GraphSize {
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
}

/// 2D rectangle in graph space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphRect {
    pub pos: GraphPos,
    pub size: GraphSize,
}

impl GraphRect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            pos: GraphPos::new(x, y),
            size: GraphSize::new(w, h),
        }
    }

    pub fn contains(&self, p: GraphPos) -> bool {
        p.x >= self.pos.x
            && p.x <= self.pos.x + self.size.width
            && p.y >= self.pos.y
            && p.y <= self.pos.y + self.size.height
    }

    pub fn center(&self) -> GraphPos {
        GraphPos::new(
            self.pos.x + self.size.width * 0.5,
            self.pos.y + self.size.height * 0.5,
        )
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.pos.x < other.pos.x + other.size.width
            && self.pos.x + self.size.width > other.pos.x
            && self.pos.y < other.pos.y + other.size.height
            && self.pos.y + self.size.height > other.pos.y
    }

    pub fn expand(&self, amount: f32) -> Self {
        Self::new(
            self.pos.x - amount,
            self.pos.y - amount,
            self.size.width + amount * 2.0,
            self.size.height + amount * 2.0,
        )
    }

    pub fn union(&self, other: &Self) -> Self {
        let min_x = self.pos.x.min(other.pos.x);
        let min_y = self.pos.y.min(other.pos.y);
        let max_x = (self.pos.x + self.size.width).max(other.pos.x + other.size.width);
        let max_y = (self.pos.y + self.size.height).max(other.pos.y + other.size.height);
        Self::new(min_x, min_y, max_x - min_x, max_y - min_y)
    }
}

// ---------------------------------------------------------------------------
// Pin (input/output port on a node)
// ---------------------------------------------------------------------------

/// Unique pin identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PinId(pub u64);

/// Direction of a pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinDirectionV2 {
    Input,
    Output,
}

/// Data type carried by a pin (for validation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PinDataTypeV2 {
    Float,
    Vec2,
    Vec3,
    Vec4,
    Color,
    Texture,
    Bool,
    Int,
    String,
    Any,
    Execution,
    Custom(u32),
}

impl PinDataTypeV2 {
    /// Check if this type can connect to another.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        if *self == Self::Any || *other == Self::Any {
            return true;
        }
        self == other
    }

    /// Get the color for this pin type (RGBA).
    pub fn color(&self) -> [f32; 4] {
        match self {
            Self::Float => [0.5, 0.8, 0.2, 1.0],
            Self::Vec2 => [0.2, 0.8, 0.5, 1.0],
            Self::Vec3 => [0.8, 0.8, 0.2, 1.0],
            Self::Vec4 => [0.8, 0.6, 0.2, 1.0],
            Self::Color => [0.8, 0.2, 0.2, 1.0],
            Self::Texture => [0.8, 0.2, 0.8, 1.0],
            Self::Bool => [0.8, 0.2, 0.2, 1.0],
            Self::Int => [0.2, 0.5, 0.8, 1.0],
            Self::String => [0.8, 0.5, 0.8, 1.0],
            Self::Any => [0.7, 0.7, 0.7, 1.0],
            Self::Execution => [1.0, 1.0, 1.0, 1.0],
            Self::Custom(_) => [0.5, 0.5, 0.5, 1.0],
        }
    }

    /// Get the display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Float => "Float",
            Self::Vec2 => "Vec2",
            Self::Vec3 => "Vec3",
            Self::Vec4 => "Vec4",
            Self::Color => "Color",
            Self::Texture => "Texture",
            Self::Bool => "Bool",
            Self::Int => "Int",
            Self::String => "String",
            Self::Any => "Any",
            Self::Execution => "Exec",
            Self::Custom(_) => "Custom",
        }
    }
}

/// A pin on a graph node.
#[derive(Debug, Clone)]
pub struct GraphPinV2 {
    pub id: PinId,
    pub name: String,
    pub direction: PinDirectionV2,
    pub data_type: PinDataTypeV2,
    pub position: GraphPos,
    pub connected: bool,
    pub default_value: Option<String>,
    pub tooltip: String,
    pub hidden: bool,
}

impl GraphPinV2 {
    pub fn input(id: PinId, name: impl Into<String>, data_type: PinDataTypeV2) -> Self {
        Self {
            id,
            name: name.into(),
            direction: PinDirectionV2::Input,
            data_type,
            position: GraphPos::ZERO,
            connected: false,
            default_value: None,
            tooltip: String::new(),
            hidden: false,
        }
    }

    pub fn output(id: PinId, name: impl Into<String>, data_type: PinDataTypeV2) -> Self {
        Self {
            id,
            name: name.into(),
            direction: PinDirectionV2::Output,
            data_type,
            position: GraphPos::ZERO,
            connected: false,
            default_value: None,
            tooltip: String::new(),
            hidden: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Graph node
// ---------------------------------------------------------------------------

/// Unique node identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphNodeId(pub u64);

/// Category of a graph node for palette grouping.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeCategory {
    pub name: String,
    pub color: [u8; 4],
    pub icon: Option<String>,
}

impl NodeCategory {
    pub fn new(name: impl Into<String>, color: [u8; 4]) -> Self {
        Self {
            name: name.into(),
            color,
            icon: None,
        }
    }
}

/// A node in the graph editor.
#[derive(Debug, Clone)]
pub struct GraphNodeV2 {
    pub id: GraphNodeId,
    pub title: String,
    pub position: GraphPos,
    pub size: GraphSize,
    pub input_pins: Vec<GraphPinV2>,
    pub output_pins: Vec<GraphPinV2>,
    pub category: Option<NodeCategory>,
    pub color: [f32; 4],
    pub selected: bool,
    pub collapsed: bool,
    pub comment: Option<String>,
    pub error: Option<String>,
    pub preview_texture: Option<u64>,
    pub user_data: HashMap<String, String>,
    pub resizable: bool,
    pub deletable: bool,
    pub movable: bool,
}

impl GraphNodeV2 {
    pub fn new(id: GraphNodeId, title: impl Into<String>, position: GraphPos) -> Self {
        Self {
            id,
            title: title.into(),
            position,
            size: GraphSize::new(200.0, 100.0),
            input_pins: Vec::new(),
            output_pins: Vec::new(),
            category: None,
            color: [0.2, 0.2, 0.2, 1.0],
            selected: false,
            collapsed: false,
            comment: None,
            error: None,
            preview_texture: None,
            user_data: HashMap::new(),
            resizable: false,
            deletable: true,
            movable: true,
        }
    }

    pub fn add_input(&mut self, pin: GraphPinV2) {
        self.input_pins.push(pin);
        self.recalculate_size();
    }

    pub fn add_output(&mut self, pin: GraphPinV2) {
        self.output_pins.push(pin);
        self.recalculate_size();
    }

    pub fn rect(&self) -> GraphRect {
        GraphRect::new(
            self.position.x,
            self.position.y,
            self.size.width,
            self.size.height,
        )
    }

    fn recalculate_size(&mut self) {
        let pin_count = self.input_pins.len().max(self.output_pins.len()) as f32;
        let min_height = 40.0 + pin_count * 24.0;
        self.size.height = min_height.max(self.size.height);
    }

    pub fn find_pin(&self, pin_id: PinId) -> Option<&GraphPinV2> {
        self.input_pins
            .iter()
            .chain(self.output_pins.iter())
            .find(|p| p.id == pin_id)
    }

    pub fn find_pin_mut(&mut self, pin_id: PinId) -> Option<&mut GraphPinV2> {
        self.input_pins
            .iter_mut()
            .chain(self.output_pins.iter_mut())
            .find(|p| p.id == pin_id)
    }

    pub fn pin_position(&self, pin_id: PinId) -> Option<GraphPos> {
        let total_inputs = self.input_pins.len() as f32;
        let total_outputs = self.output_pins.len() as f32;

        for (i, pin) in self.input_pins.iter().enumerate() {
            if pin.id == pin_id {
                let y_offset = 30.0 + (i as f32 + 0.5) * (self.size.height - 30.0) / total_inputs;
                return Some(GraphPos::new(self.position.x, self.position.y + y_offset));
            }
        }

        for (i, pin) in self.output_pins.iter().enumerate() {
            if pin.id == pin_id {
                let y_offset =
                    30.0 + (i as f32 + 0.5) * (self.size.height - 30.0) / total_outputs;
                return Some(GraphPos::new(
                    self.position.x + self.size.width,
                    self.position.y + y_offset,
                ));
            }
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Wire (connection)
// ---------------------------------------------------------------------------

/// Unique wire identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WireId(pub u64);

/// A wire connecting two pins.
#[derive(Debug, Clone)]
pub struct GraphWireV2 {
    pub id: WireId,
    pub source_node: GraphNodeId,
    pub source_pin: PinId,
    pub target_node: GraphNodeId,
    pub target_pin: PinId,
    pub color: [f32; 4],
    pub thickness: f32,
    pub reroute_points: Vec<GraphPos>,
    pub selected: bool,
}

impl GraphWireV2 {
    pub fn new(
        id: WireId,
        source_node: GraphNodeId,
        source_pin: PinId,
        target_node: GraphNodeId,
        target_pin: PinId,
    ) -> Self {
        Self {
            id,
            source_node,
            source_pin,
            target_node,
            target_pin,
            color: [1.0, 1.0, 1.0, 0.8],
            thickness: 2.0,
            reroute_points: Vec::new(),
            selected: false,
        }
    }

    pub fn add_reroute(&mut self, pos: GraphPos) {
        self.reroute_points.push(pos);
    }

    /// Get the Bezier control points for rendering.
    pub fn bezier_points(
        &self,
        source_pos: GraphPos,
        target_pos: GraphPos,
    ) -> (GraphPos, GraphPos, GraphPos, GraphPos) {
        let dx = (target_pos.x - source_pos.x).abs() * 0.5;
        let cp1 = GraphPos::new(source_pos.x + dx, source_pos.y);
        let cp2 = GraphPos::new(target_pos.x - dx, target_pos.y);
        (source_pos, cp1, cp2, target_pos)
    }
}

// ---------------------------------------------------------------------------
// Comment box
// ---------------------------------------------------------------------------

/// A comment box / annotation in the graph.
#[derive(Debug, Clone)]
pub struct CommentBox {
    pub id: u64,
    pub rect: GraphRect,
    pub text: String,
    pub color: [f32; 4],
    pub font_size: f32,
    pub selected: bool,
    pub move_with_nodes: bool,
    pub contained_nodes: Vec<GraphNodeId>,
    pub z_order: i32,
}

impl CommentBox {
    pub fn new(id: u64, rect: GraphRect, text: impl Into<String>) -> Self {
        Self {
            id,
            rect,
            text: text.into(),
            color: [0.1, 0.1, 0.1, 0.3],
            font_size: 16.0,
            selected: false,
            move_with_nodes: true,
            contained_nodes: Vec::new(),
            z_order: -1,
        }
    }

    pub fn update_contained_nodes(&mut self, nodes: &HashMap<GraphNodeId, GraphNodeV2>) {
        self.contained_nodes.clear();
        for (&id, node) in nodes {
            if self.rect.contains(node.position) {
                self.contained_nodes.push(id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Minimap
// ---------------------------------------------------------------------------

/// Minimap state for overview navigation.
#[derive(Debug, Clone)]
pub struct Minimap {
    pub rect: GraphRect,
    pub visible: bool,
    pub world_bounds: GraphRect,
    pub view_rect: GraphRect,
    pub opacity: f32,
    pub background_color: [f32; 4],
    pub node_color: [f32; 4],
    pub view_color: [f32; 4],
}

impl Minimap {
    pub fn new() -> Self {
        Self {
            rect: GraphRect::new(0.0, 0.0, 200.0, 150.0),
            visible: true,
            world_bounds: GraphRect::new(-1000.0, -1000.0, 2000.0, 2000.0),
            view_rect: GraphRect::new(-500.0, -500.0, 1000.0, 1000.0),
            opacity: 0.7,
            background_color: [0.05, 0.05, 0.05, 0.7],
            node_color: [0.3, 0.6, 0.3, 0.8],
            view_color: [1.0, 1.0, 1.0, 0.3],
        }
    }

    pub fn world_to_minimap(&self, world_pos: GraphPos) -> GraphPos {
        let nx = (world_pos.x - self.world_bounds.pos.x) / self.world_bounds.size.width;
        let ny = (world_pos.y - self.world_bounds.pos.y) / self.world_bounds.size.height;
        GraphPos::new(
            self.rect.pos.x + nx * self.rect.size.width,
            self.rect.pos.y + ny * self.rect.size.height,
        )
    }

    pub fn minimap_to_world(&self, minimap_pos: GraphPos) -> GraphPos {
        let nx = (minimap_pos.x - self.rect.pos.x) / self.rect.size.width;
        let ny = (minimap_pos.y - self.rect.pos.y) / self.rect.size.height;
        GraphPos::new(
            self.world_bounds.pos.x + nx * self.world_bounds.size.width,
            self.world_bounds.pos.y + ny * self.world_bounds.size.height,
        )
    }

    pub fn update_world_bounds(&mut self, nodes: &HashMap<GraphNodeId, GraphNodeV2>) {
        if nodes.is_empty() {
            return;
        }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for node in nodes.values() {
            min_x = min_x.min(node.position.x);
            min_y = min_y.min(node.position.y);
            max_x = max_x.max(node.position.x + node.size.width);
            max_y = max_y.max(node.position.y + node.size.height);
        }

        let padding = 100.0;
        self.world_bounds = GraphRect::new(
            min_x - padding,
            min_y - padding,
            (max_x - min_x) + padding * 2.0,
            (max_y - min_y) + padding * 2.0,
        );
    }
}

impl Default for Minimap {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Connection validation
// ---------------------------------------------------------------------------

/// Result of a connection validation check.
#[derive(Debug, Clone)]
pub enum ConnectionValidation {
    Valid,
    Invalid(String),
    ValidWithConversion(String),
}

impl ConnectionValidation {
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid | Self::ValidWithConversion(_))
    }
}

/// Validates whether a connection between two pins is allowed.
pub fn validate_connection(
    source: &GraphPinV2,
    target: &GraphPinV2,
    existing_wires: &[GraphWireV2],
) -> ConnectionValidation {
    // Must connect output to input.
    if source.direction == target.direction {
        return ConnectionValidation::Invalid(
            "Cannot connect two pins of the same direction".to_string(),
        );
    }

    // Type compatibility.
    if !source.data_type.is_compatible_with(&target.data_type) {
        return ConnectionValidation::Invalid(format!(
            "Type mismatch: {} -> {}",
            source.data_type.display_name(),
            target.data_type.display_name()
        ));
    }

    // Check if the target input already has a connection (inputs allow one wire).
    if target.direction == PinDirectionV2::Input {
        let already_connected = existing_wires
            .iter()
            .any(|w| w.target_pin == target.id);
        if already_connected {
            return ConnectionValidation::Invalid(
                "Input pin already has a connection".to_string(),
            );
        }
    }

    ConnectionValidation::Valid
}

// ---------------------------------------------------------------------------
// Force-directed auto-layout
// ---------------------------------------------------------------------------

/// Configuration for force-directed auto-layout.
#[derive(Debug, Clone)]
pub struct ForceDirectedConfig {
    pub repulsion_strength: f32,
    pub attraction_strength: f32,
    pub damping: f32,
    pub ideal_edge_length: f32,
    pub max_iterations: u32,
    pub convergence_threshold: f32,
    pub gravity_strength: f32,
    pub gravity_center: GraphPos,
}

impl Default for ForceDirectedConfig {
    fn default() -> Self {
        Self {
            repulsion_strength: 5000.0,
            attraction_strength: 0.01,
            damping: 0.95,
            ideal_edge_length: 200.0,
            max_iterations: 500,
            convergence_threshold: 0.1,
            gravity_strength: 0.001,
            gravity_center: GraphPos::ZERO,
        }
    }
}

/// Run force-directed layout on a set of nodes and wires.
pub fn force_directed_layout(
    nodes: &mut HashMap<GraphNodeId, GraphNodeV2>,
    wires: &[GraphWireV2],
    config: &ForceDirectedConfig,
) -> u32 {
    let node_ids: Vec<GraphNodeId> = nodes.keys().copied().collect();
    let mut velocities: HashMap<GraphNodeId, GraphPos> = HashMap::new();
    for &id in &node_ids {
        velocities.insert(id, GraphPos::ZERO);
    }

    let mut iterations = 0;

    for _ in 0..config.max_iterations {
        iterations += 1;
        let mut forces: HashMap<GraphNodeId, GraphPos> = HashMap::new();
        for &id in &node_ids {
            forces.insert(id, GraphPos::ZERO);
        }

        // Repulsion between all node pairs.
        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                let a = node_ids[i];
                let b = node_ids[j];
                let pa = nodes[&a].position;
                let pb = nodes[&b].position;
                let dx = pa.x - pb.x;
                let dy = pa.y - pb.y;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                let force = config.repulsion_strength / (dist * dist);
                let fx = dx / dist * force;
                let fy = dy / dist * force;

                forces.get_mut(&a).unwrap().x += fx;
                forces.get_mut(&a).unwrap().y += fy;
                forces.get_mut(&b).unwrap().x -= fx;
                forces.get_mut(&b).unwrap().y -= fy;
            }
        }

        // Attraction along wires.
        for wire in wires {
            if let (Some(src), Some(dst)) =
                (nodes.get(&wire.source_node), nodes.get(&wire.target_node))
            {
                let dx = dst.position.x - src.position.x;
                let dy = dst.position.y - src.position.y;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                let force =
                    config.attraction_strength * (dist - config.ideal_edge_length);
                let fx = dx / dist * force;
                let fy = dy / dist * force;

                if let Some(f) = forces.get_mut(&wire.source_node) {
                    f.x += fx;
                    f.y += fy;
                }
                if let Some(f) = forces.get_mut(&wire.target_node) {
                    f.x -= fx;
                    f.y -= fy;
                }
            }
        }

        // Gravity.
        for &id in &node_ids {
            let pos = nodes[&id].position;
            if let Some(f) = forces.get_mut(&id) {
                f.x += (config.gravity_center.x - pos.x) * config.gravity_strength;
                f.y += (config.gravity_center.y - pos.y) * config.gravity_strength;
            }
        }

        // Apply forces.
        let mut total_displacement = 0.0f32;
        for &id in &node_ids {
            let f = forces[&id];
            let vel = velocities.get_mut(&id).unwrap();
            vel.x = (vel.x + f.x) * config.damping;
            vel.y = (vel.y + f.y) * config.damping;

            let node = nodes.get_mut(&id).unwrap();
            if node.movable {
                node.position.x += vel.x;
                node.position.y += vel.y;
            }

            total_displacement += (vel.x * vel.x + vel.y * vel.y).sqrt();
        }

        if total_displacement < config.convergence_threshold {
            break;
        }
    }

    iterations
}

// ---------------------------------------------------------------------------
// Zoom levels
// ---------------------------------------------------------------------------

/// Zoom level presets.
pub const ZOOM_PRESETS: &[(f32, &str)] = &[
    (0.25, "25%"),
    (0.5, "50%"),
    (0.75, "75%"),
    (1.0, "100%"),
    (1.5, "150%"),
    (2.0, "200%"),
    (3.0, "300%"),
];

/// Find the next zoom preset (for zoom-in).
pub fn next_zoom_preset(current: f32) -> f32 {
    for &(scale, _) in ZOOM_PRESETS {
        if scale > current + 0.01 {
            return scale;
        }
    }
    current
}

/// Find the previous zoom preset (for zoom-out).
pub fn prev_zoom_preset(current: f32) -> f32 {
    for &(scale, _) in ZOOM_PRESETS.iter().rev() {
        if scale < current - 0.01 {
            return scale;
        }
    }
    current
}

// ---------------------------------------------------------------------------
// Graph editor event
// ---------------------------------------------------------------------------

/// Events emitted by the graph editor.
#[derive(Debug, Clone)]
pub enum GraphEditorEvent {
    NodeAdded(GraphNodeId),
    NodeRemoved(GraphNodeId),
    NodeMoved(GraphNodeId, GraphPos),
    NodeSelected(GraphNodeId),
    NodeDeselected(GraphNodeId),
    WireAdded(WireId),
    WireRemoved(WireId),
    CommentAdded(u64),
    CommentRemoved(u64),
    ZoomChanged(f32),
    PanChanged(GraphPos),
    SelectionCleared,
    LayoutCompleted(u32),
}

// ---------------------------------------------------------------------------
// Graph editor state
// ---------------------------------------------------------------------------

/// Main graph editor state.
pub struct GraphEditorState {
    pub nodes: HashMap<GraphNodeId, GraphNodeV2>,
    pub wires: Vec<GraphWireV2>,
    pub comments: Vec<CommentBox>,
    pub minimap: Minimap,
    pub zoom: f32,
    pub pan_offset: GraphPos,
    pub selected_nodes: Vec<GraphNodeId>,
    pub selected_wires: Vec<WireId>,
    pub events: Vec<GraphEditorEvent>,
    pub next_node_id: u64,
    pub next_wire_id: u64,
    pub next_pin_id: u64,
    pub next_comment_id: u64,
    pub grid_size: f32,
    pub snap_to_grid: bool,
    pub show_grid: bool,
    pub show_minimap: bool,
    pub show_wire_names: bool,
    pub node_categories: Vec<NodeCategory>,
    pub background_color: [f32; 4],
    pub grid_color: [f32; 4],
    pub selection_color: [f32; 4],
}

impl GraphEditorState {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            wires: Vec::new(),
            comments: Vec::new(),
            minimap: Minimap::new(),
            zoom: 1.0,
            pan_offset: GraphPos::ZERO,
            selected_nodes: Vec::new(),
            selected_wires: Vec::new(),
            events: Vec::new(),
            next_node_id: 1,
            next_wire_id: 1,
            next_pin_id: 1,
            next_comment_id: 1,
            grid_size: 16.0,
            snap_to_grid: true,
            show_grid: true,
            show_minimap: true,
            show_wire_names: false,
            node_categories: Vec::new(),
            background_color: [0.12, 0.12, 0.12, 1.0],
            grid_color: [0.15, 0.15, 0.15, 1.0],
            selection_color: [0.2, 0.5, 1.0, 0.3],
        }
    }

    pub fn alloc_node_id(&mut self) -> GraphNodeId {
        let id = GraphNodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    pub fn alloc_wire_id(&mut self) -> WireId {
        let id = WireId(self.next_wire_id);
        self.next_wire_id += 1;
        id
    }

    pub fn alloc_pin_id(&mut self) -> PinId {
        let id = PinId(self.next_pin_id);
        self.next_pin_id += 1;
        id
    }

    pub fn add_node(&mut self, mut node: GraphNodeV2) -> GraphNodeId {
        let id = node.id;
        if self.snap_to_grid {
            node.position.x = (node.position.x / self.grid_size).round() * self.grid_size;
            node.position.y = (node.position.y / self.grid_size).round() * self.grid_size;
        }
        self.nodes.insert(id, node);
        self.events.push(GraphEditorEvent::NodeAdded(id));
        id
    }

    pub fn remove_node(&mut self, id: GraphNodeId) -> bool {
        if self.nodes.remove(&id).is_some() {
            self.wires
                .retain(|w| w.source_node != id && w.target_node != id);
            self.selected_nodes.retain(|&n| n != id);
            self.events.push(GraphEditorEvent::NodeRemoved(id));
            true
        } else {
            false
        }
    }

    pub fn add_wire(&mut self, wire: GraphWireV2) -> WireId {
        let id = wire.id;
        if let Some(node) = self.nodes.get_mut(&wire.source_node) {
            if let Some(pin) = node.find_pin_mut(wire.source_pin) {
                pin.connected = true;
            }
        }
        if let Some(node) = self.nodes.get_mut(&wire.target_node) {
            if let Some(pin) = node.find_pin_mut(wire.target_pin) {
                pin.connected = true;
            }
        }
        self.wires.push(wire);
        self.events.push(GraphEditorEvent::WireAdded(id));
        id
    }

    pub fn remove_wire(&mut self, id: WireId) -> bool {
        if let Some(pos) = self.wires.iter().position(|w| w.id == id) {
            let wire = self.wires.remove(pos);
            self.events.push(GraphEditorEvent::WireRemoved(id));
            let src_still = self.wires.iter().any(|w| w.source_pin == wire.source_pin);
            let dst_still = self.wires.iter().any(|w| w.target_pin == wire.target_pin);
            if let Some(node) = self.nodes.get_mut(&wire.source_node) {
                if let Some(pin) = node.find_pin_mut(wire.source_pin) {
                    pin.connected = src_still;
                }
            }
            if let Some(node) = self.nodes.get_mut(&wire.target_node) {
                if let Some(pin) = node.find_pin_mut(wire.target_pin) {
                    pin.connected = dst_still;
                }
            }
            true
        } else {
            false
        }
    }

    pub fn select_node(&mut self, id: GraphNodeId) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.selected = true;
            if !self.selected_nodes.contains(&id) {
                self.selected_nodes.push(id);
            }
            self.events.push(GraphEditorEvent::NodeSelected(id));
        }
    }

    pub fn deselect_all(&mut self) {
        for node in self.nodes.values_mut() {
            node.selected = false;
        }
        for wire in &mut self.wires {
            wire.selected = false;
        }
        self.selected_nodes.clear();
        self.selected_wires.clear();
        self.events.push(GraphEditorEvent::SelectionCleared);
    }

    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom.clamp(0.1, 5.0);
        self.events.push(GraphEditorEvent::ZoomChanged(self.zoom));
    }

    pub fn zoom_in(&mut self) {
        self.set_zoom(self.zoom * 1.2);
    }

    pub fn zoom_out(&mut self) {
        self.set_zoom(self.zoom / 1.2);
    }

    pub fn zoom_to_fit(&mut self) {
        self.minimap.update_world_bounds(&self.nodes);
        let bounds = self.minimap.world_bounds;
        self.pan_offset = bounds.center().scale(-1.0);
        let zoom_x = 1000.0 / bounds.size.width.max(1.0);
        let zoom_y = 800.0 / bounds.size.height.max(1.0);
        self.set_zoom(zoom_x.min(zoom_y).min(1.0));
    }

    pub fn screen_to_graph(&self, screen_pos: GraphPos) -> GraphPos {
        GraphPos::new(
            (screen_pos.x - self.pan_offset.x) / self.zoom,
            (screen_pos.y - self.pan_offset.y) / self.zoom,
        )
    }

    pub fn graph_to_screen(&self, graph_pos: GraphPos) -> GraphPos {
        GraphPos::new(
            graph_pos.x * self.zoom + self.pan_offset.x,
            graph_pos.y * self.zoom + self.pan_offset.y,
        )
    }

    pub fn node_at_position(&self, pos: GraphPos) -> Option<GraphNodeId> {
        self.nodes
            .iter()
            .find(|(_, node)| node.rect().contains(pos))
            .map(|(&id, _)| id)
    }

    pub fn nodes_in_rect(&self, rect: GraphRect) -> Vec<GraphNodeId> {
        self.nodes
            .iter()
            .filter(|(_, node)| rect.intersects(&node.rect()))
            .map(|(&id, _)| id)
            .collect()
    }

    pub fn delete_selected(&mut self) {
        let nodes: Vec<GraphNodeId> = self.selected_nodes.clone();
        for id in nodes {
            self.remove_node(id);
        }
        let wires: Vec<WireId> = self.selected_wires.clone();
        for id in wires {
            self.remove_wire(id);
        }
    }

    pub fn add_comment(&mut self, rect: GraphRect, text: impl Into<String>) -> u64 {
        let id = self.next_comment_id;
        self.next_comment_id += 1;
        self.comments.push(CommentBox::new(id, rect, text));
        self.events.push(GraphEditorEvent::CommentAdded(id));
        id
    }

    pub fn auto_layout(&mut self) {
        let config = ForceDirectedConfig::default();
        let iterations = force_directed_layout(&mut self.nodes, &self.wires, &config);
        self.events
            .push(GraphEditorEvent::LayoutCompleted(iterations));
    }

    pub fn drain_events(&mut self) -> Vec<GraphEditorEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn wire_count(&self) -> usize {
        self.wires.len()
    }

    /// Duplicate selected nodes with offset.
    pub fn duplicate_selected(&mut self, offset: GraphPos) -> Vec<GraphNodeId> {
        let selected: Vec<GraphNodeId> = self.selected_nodes.clone();
        let mut new_ids = Vec::new();
        let mut id_map: HashMap<GraphNodeId, GraphNodeId> = HashMap::new();

        for old_id in &selected {
            if let Some(old_node) = self.nodes.get(old_id) {
                let new_id = self.alloc_node_id();
                let mut new_node = old_node.clone();
                new_node.id = new_id;
                new_node.position = old_node.position.add(&offset);
                new_node.selected = false;

                // Remap pin IDs.
                for pin in new_node.input_pins.iter_mut() {
                    pin.id = self.alloc_pin_id();
                    pin.connected = false;
                }
                for pin in new_node.output_pins.iter_mut() {
                    pin.id = self.alloc_pin_id();
                    pin.connected = false;
                }

                self.nodes.insert(new_id, new_node);
                id_map.insert(*old_id, new_id);
                new_ids.push(new_id);
            }
        }

        new_ids
    }

    /// Get all connections for a node.
    pub fn wires_for_node(&self, node_id: GraphNodeId) -> Vec<&GraphWireV2> {
        self.wires
            .iter()
            .filter(|w| w.source_node == node_id || w.target_node == node_id)
            .collect()
    }

    /// Calculate the bounds of all nodes.
    pub fn bounds(&self) -> Option<GraphRect> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for node in self.nodes.values() {
            min_x = min_x.min(node.position.x);
            min_y = min_y.min(node.position.y);
            max_x = max_x.max(node.position.x + node.size.width);
            max_y = max_y.max(node.position.y + node.size.height);
        }

        Some(GraphRect::new(min_x, min_y, max_x - min_x, max_y - min_y))
    }
}

impl Default for GraphEditorState {
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
    fn graph_editor_basic() {
        let mut editor = GraphEditorState::new();
        let id = editor.alloc_node_id();
        let node = GraphNodeV2::new(id, "Test", GraphPos::new(100.0, 200.0));
        editor.add_node(node);
        assert_eq!(editor.node_count(), 1);

        editor.remove_node(id);
        assert_eq!(editor.node_count(), 0);
    }

    #[test]
    fn wire_connection() {
        let mut editor = GraphEditorState::new();
        let n1 = editor.alloc_node_id();
        let n2 = editor.alloc_node_id();
        let p1 = editor.alloc_pin_id();
        let p2 = editor.alloc_pin_id();

        let mut node1 = GraphNodeV2::new(n1, "A", GraphPos::ZERO);
        node1.add_output(GraphPinV2::output(p1, "Out", PinDataTypeV2::Float));
        editor.add_node(node1);

        let mut node2 = GraphNodeV2::new(n2, "B", GraphPos::new(300.0, 0.0));
        node2.add_input(GraphPinV2::input(p2, "In", PinDataTypeV2::Float));
        editor.add_node(node2);

        let wid = editor.alloc_wire_id();
        let wire = GraphWireV2::new(wid, n1, p1, n2, p2);
        editor.add_wire(wire);
        assert_eq!(editor.wire_count(), 1);
    }

    #[test]
    fn zoom_clamp() {
        let mut editor = GraphEditorState::new();
        editor.set_zoom(100.0);
        assert!(editor.zoom <= 5.0);
        editor.set_zoom(0.01);
        assert!(editor.zoom >= 0.1);
    }

    #[test]
    fn pin_compatibility() {
        assert!(PinDataTypeV2::Float.is_compatible_with(&PinDataTypeV2::Float));
        assert!(PinDataTypeV2::Any.is_compatible_with(&PinDataTypeV2::Float));
        assert!(!PinDataTypeV2::Float.is_compatible_with(&PinDataTypeV2::Texture));
    }

    #[test]
    fn graph_rect_operations() {
        let a = GraphRect::new(0.0, 0.0, 10.0, 10.0);
        let b = GraphRect::new(5.0, 5.0, 10.0, 10.0);
        assert!(a.intersects(&b));
        assert!(a.contains(GraphPos::new(5.0, 5.0)));
    }

    #[test]
    fn connection_validation() {
        let src = GraphPinV2::output(PinId(1), "Out", PinDataTypeV2::Float);
        let dst = GraphPinV2::input(PinId(2), "In", PinDataTypeV2::Float);
        let result = validate_connection(&src, &dst, &[]);
        assert!(result.is_valid());

        let bad_dst = GraphPinV2::input(PinId(3), "In", PinDataTypeV2::Texture);
        let result = validate_connection(&src, &bad_dst, &[]);
        assert!(!result.is_valid());
    }
}
