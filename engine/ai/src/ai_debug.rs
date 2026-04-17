// engine/ai/src/ai_debug.rs
//
// AI debug visualization for the Genovo engine.
//
// Provides debug drawing utilities for visualizing AI state:
//
// - **Perception cones** -- Draw vision/hearing sensor cones.
// - **Awareness levels** -- Color-coded awareness state visualization.
// - **Behavior tree node** -- Show current BT node and path.
// - **Navigation path** -- Draw the current nav path with waypoints.
// - **Steering vectors** -- Visualize steering forces and velocities.
// - **Influence map values** -- Heat map overlay for influence maps.
// - **Blackboard contents** -- Text overlay of blackboard key-value pairs.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_DEBUG_PRIMITIVES: usize = 10000;
const MAX_TEXT_ENTRIES: usize = 256;

// ---------------------------------------------------------------------------
// Debug draw primitives
// ---------------------------------------------------------------------------

/// Color for debug drawing (RGBA, 0..1).
#[derive(Debug, Clone, Copy)]
pub struct DebugColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl DebugColor {
    pub const RED: Self = Self { r: 1.0, g: 0.0, b: 0.0, a: 0.8 };
    pub const GREEN: Self = Self { r: 0.0, g: 1.0, b: 0.0, a: 0.8 };
    pub const BLUE: Self = Self { r: 0.0, g: 0.0, b: 1.0, a: 0.8 };
    pub const YELLOW: Self = Self { r: 1.0, g: 1.0, b: 0.0, a: 0.8 };
    pub const CYAN: Self = Self { r: 0.0, g: 1.0, b: 1.0, a: 0.8 };
    pub const MAGENTA: Self = Self { r: 1.0, g: 0.0, b: 1.0, a: 0.8 };
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 0.8 };
    pub const ORANGE: Self = Self { r: 1.0, g: 0.5, b: 0.0, a: 0.8 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn with_alpha(mut self, a: f32) -> Self {
        self.a = a;
        self
    }

    pub fn to_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Lerp between two colors.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}

/// A debug draw primitive.
#[derive(Debug, Clone)]
pub enum DebugPrimitive {
    /// Line segment.
    Line { start: [f32; 3], end: [f32; 3], color: DebugColor, thickness: f32 },
    /// Circle (Y-up plane).
    Circle { center: [f32; 3], radius: f32, color: DebugColor, segments: u32 },
    /// Sphere wireframe.
    Sphere { center: [f32; 3], radius: f32, color: DebugColor },
    /// Cone (for perception).
    Cone { origin: [f32; 3], direction: [f32; 3], angle: f32, length: f32, color: DebugColor },
    /// Arrow (line with arrowhead).
    Arrow { start: [f32; 3], end: [f32; 3], color: DebugColor, head_size: f32 },
    /// Box wireframe.
    Box { center: [f32; 3], half_extents: [f32; 3], color: DebugColor },
    /// Filled quad (for influence map cells).
    Quad { center: [f32; 3], size: [f32; 2], color: DebugColor },
    /// Path (polyline).
    Path { points: Vec<[f32; 3]>, color: DebugColor, thickness: f32, closed: bool },
    /// Text label in 3D space.
    Text3D { position: [f32; 3], text: String, color: DebugColor, size: f32 },
}

/// A 2D screen-space text entry.
#[derive(Debug, Clone)]
pub struct DebugTextEntry {
    pub text: String,
    pub position: [f32; 2],
    pub color: DebugColor,
    pub size: f32,
    pub category: String,
}

// ---------------------------------------------------------------------------
// AI awareness visualization
// ---------------------------------------------------------------------------

/// Awareness level for color coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AwarenessVisualLevel {
    Unaware,
    Suspicious,
    Alerted,
    Combat,
    Searching,
    Lost,
}

impl AwarenessVisualLevel {
    /// Get the debug color for this awareness level.
    pub fn color(&self) -> DebugColor {
        match self {
            Self::Unaware => DebugColor::GREEN,
            Self::Suspicious => DebugColor::YELLOW,
            Self::Alerted => DebugColor::ORANGE,
            Self::Combat => DebugColor::RED,
            Self::Searching => DebugColor::MAGENTA,
            Self::Lost => DebugColor::CYAN,
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Self::Unaware => "Unaware",
            Self::Suspicious => "Suspicious",
            Self::Alerted => "ALERT",
            Self::Combat => "COMBAT",
            Self::Searching => "Searching",
            Self::Lost => "Lost Contact",
        }
    }
}

// ---------------------------------------------------------------------------
// Behavior tree debug info
// ---------------------------------------------------------------------------

/// Debug info for a behavior tree node.
#[derive(Debug, Clone)]
pub struct BtNodeDebugInfo {
    /// Node name.
    pub name: String,
    /// Node type (Sequence, Selector, Action, etc.).
    pub node_type: String,
    /// Current status.
    pub status: String,
    /// Depth in the tree (for indentation).
    pub depth: u32,
    /// Whether this node is currently active.
    pub active: bool,
    /// Execution time of this node (ms).
    pub exec_time_ms: f32,
}

/// Full behavior tree debug state.
#[derive(Debug, Clone)]
pub struct BtDebugState {
    /// Agent ID.
    pub agent_id: u32,
    /// Tree name.
    pub tree_name: String,
    /// All nodes with their current state.
    pub nodes: Vec<BtNodeDebugInfo>,
    /// The path of currently active nodes (root to leaf).
    pub active_path: Vec<String>,
    /// Total tree evaluation time.
    pub eval_time_ms: f32,
}

// ---------------------------------------------------------------------------
// Blackboard debug
// ---------------------------------------------------------------------------

/// Debug view of a blackboard entry.
#[derive(Debug, Clone)]
pub struct BlackboardDebugEntry {
    pub key: String,
    pub value: String,
    pub type_name: String,
    pub last_modified_frame: u64,
}

// ---------------------------------------------------------------------------
// Steering debug
// ---------------------------------------------------------------------------

/// Debug info for steering behaviors.
#[derive(Debug, Clone)]
pub struct SteeringDebugInfo {
    pub agent_position: [f32; 3],
    pub velocity: [f32; 3],
    pub desired_velocity: [f32; 3],
    pub steering_force: [f32; 3],
    pub individual_forces: Vec<(String, [f32; 3])>,
}

// ---------------------------------------------------------------------------
// AI debug settings
// ---------------------------------------------------------------------------

/// Configuration for which AI debug features to show.
#[derive(Debug, Clone)]
pub struct AiDebugSettings {
    pub enabled: bool,
    pub show_perception_cones: bool,
    pub show_awareness_levels: bool,
    pub show_behavior_tree: bool,
    pub show_navigation_path: bool,
    pub show_steering_vectors: bool,
    pub show_influence_map: bool,
    pub show_blackboard: bool,
    pub show_target_info: bool,
    pub show_group_info: bool,
    pub selected_agent: Option<u32>,
    pub perception_cone_alpha: f32,
    pub path_color: DebugColor,
    pub influence_map_layer: String,
    pub text_scale: f32,
}

impl Default for AiDebugSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            show_perception_cones: true,
            show_awareness_levels: true,
            show_behavior_tree: true,
            show_navigation_path: true,
            show_steering_vectors: true,
            show_influence_map: false,
            show_blackboard: true,
            show_target_info: true,
            show_group_info: true,
            selected_agent: None,
            perception_cone_alpha: 0.15,
            path_color: DebugColor::CYAN,
            influence_map_layer: "threat".to_string(),
            text_scale: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// AI debug renderer
// ---------------------------------------------------------------------------

/// Collects AI debug draw data for a frame.
pub struct AiDebugRenderer {
    settings: AiDebugSettings,
    primitives: Vec<DebugPrimitive>,
    text_entries: Vec<DebugTextEntry>,
    bt_states: Vec<BtDebugState>,
    blackboard_entries: HashMap<u32, Vec<BlackboardDebugEntry>>,
    steering_info: HashMap<u32, SteeringDebugInfo>,
    frame: u64,
}

impl AiDebugRenderer {
    pub fn new() -> Self {
        Self {
            settings: AiDebugSettings::default(),
            primitives: Vec::new(),
            text_entries: Vec::new(),
            bt_states: Vec::new(),
            blackboard_entries: HashMap::new(),
            steering_info: HashMap::new(),
            frame: 0,
        }
    }

    pub fn settings(&self) -> &AiDebugSettings {
        &self.settings
    }

    pub fn settings_mut(&mut self) -> &mut AiDebugSettings {
        &mut self.settings
    }

    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.primitives.clear();
        self.text_entries.clear();
        self.bt_states.clear();
        self.blackboard_entries.clear();
        self.steering_info.clear();
    }

    /// Draw a perception cone.
    pub fn draw_perception_cone(
        &mut self,
        origin: [f32; 3],
        direction: [f32; 3],
        half_angle_rad: f32,
        range: f32,
        awareness: AwarenessVisualLevel,
    ) {
        if !self.settings.enabled || !self.settings.show_perception_cones {
            return;
        }
        if self.primitives.len() >= MAX_DEBUG_PRIMITIVES {
            return;
        }
        let color = awareness.color().with_alpha(self.settings.perception_cone_alpha);
        self.primitives.push(DebugPrimitive::Cone {
            origin,
            direction,
            angle: half_angle_rad,
            length: range,
            color,
        });
    }

    /// Draw awareness indicator above an agent.
    pub fn draw_awareness(&mut self, position: [f32; 3], awareness: AwarenessVisualLevel) {
        if !self.settings.enabled || !self.settings.show_awareness_levels {
            return;
        }
        let label_pos = [position[0], position[1] + 2.5, position[2]];
        let color = awareness.color();

        self.primitives.push(DebugPrimitive::Circle {
            center: [position[0], position[1] + 2.0, position[2]],
            radius: 0.3,
            color,
            segments: 16,
        });
        self.primitives.push(DebugPrimitive::Text3D {
            position: label_pos,
            text: awareness.label().to_string(),
            color,
            size: self.settings.text_scale * 0.3,
        });
    }

    /// Draw a navigation path.
    pub fn draw_nav_path(&mut self, waypoints: &[[f32; 3]]) {
        if !self.settings.enabled || !self.settings.show_navigation_path || waypoints.len() < 2 {
            return;
        }
        self.primitives.push(DebugPrimitive::Path {
            points: waypoints.to_vec(),
            color: self.settings.path_color,
            thickness: 2.0,
            closed: false,
        });
        // Draw waypoint markers.
        for (i, &wp) in waypoints.iter().enumerate() {
            let color = if i == 0 { DebugColor::GREEN } else if i == waypoints.len() - 1 { DebugColor::RED } else { DebugColor::YELLOW };
            self.primitives.push(DebugPrimitive::Sphere {
                center: wp,
                radius: 0.15,
                color,
            });
        }
    }

    /// Draw steering vectors.
    pub fn draw_steering(&mut self, info: SteeringDebugInfo) {
        if !self.settings.enabled || !self.settings.show_steering_vectors {
            return;
        }
        let pos = info.agent_position;

        // Velocity arrow (blue).
        self.primitives.push(DebugPrimitive::Arrow {
            start: pos,
            end: [pos[0] + info.velocity[0], pos[1] + info.velocity[1], pos[2] + info.velocity[2]],
            color: DebugColor::BLUE,
            head_size: 0.1,
        });

        // Desired velocity (green).
        self.primitives.push(DebugPrimitive::Arrow {
            start: pos,
            end: [pos[0] + info.desired_velocity[0], pos[1] + info.desired_velocity[1], pos[2] + info.desired_velocity[2]],
            color: DebugColor::GREEN,
            head_size: 0.1,
        });

        // Steering force (red).
        self.primitives.push(DebugPrimitive::Arrow {
            start: pos,
            end: [pos[0] + info.steering_force[0], pos[1] + info.steering_force[1], pos[2] + info.steering_force[2]],
            color: DebugColor::RED,
            head_size: 0.1,
        });

        // Individual forces.
        for (name, force) in &info.individual_forces {
            self.primitives.push(DebugPrimitive::Arrow {
                start: pos,
                end: [pos[0] + force[0] * 0.5, pos[1] + force[1] * 0.5, pos[2] + force[2] * 0.5],
                color: DebugColor::WHITE.with_alpha(0.4),
                head_size: 0.05,
            });
        }

        let _ = info; // stored for UI panel
    }

    /// Draw influence map as colored grid.
    pub fn draw_influence_map(
        &mut self,
        cells: &[InfluenceCell],
        min_value: f32,
        max_value: f32,
    ) {
        if !self.settings.enabled || !self.settings.show_influence_map {
            return;
        }
        let range = (max_value - min_value).max(0.001);
        for cell in cells {
            let t = ((cell.value - min_value) / range).clamp(0.0, 1.0);
            let color = DebugColor::BLUE.lerp(&DebugColor::RED, t).with_alpha(0.3 * t);
            self.primitives.push(DebugPrimitive::Quad {
                center: cell.position,
                size: [cell.size, cell.size],
                color,
            });
        }
    }

    /// Submit behavior tree debug state.
    pub fn submit_bt_state(&mut self, state: BtDebugState) {
        if !self.settings.enabled || !self.settings.show_behavior_tree {
            return;
        }
        self.bt_states.push(state);
    }

    /// Submit blackboard contents.
    pub fn submit_blackboard(&mut self, agent_id: u32, entries: Vec<BlackboardDebugEntry>) {
        if !self.settings.enabled || !self.settings.show_blackboard {
            return;
        }
        self.blackboard_entries.insert(agent_id, entries);
    }

    /// Add a debug text entry.
    pub fn add_text(&mut self, text: &str, position: [f32; 2], color: DebugColor) {
        if self.text_entries.len() >= MAX_TEXT_ENTRIES {
            return;
        }
        self.text_entries.push(DebugTextEntry {
            text: text.to_string(),
            position,
            color,
            size: self.settings.text_scale * 14.0,
            category: String::new(),
        });
    }

    /// Get all primitives for rendering.
    pub fn primitives(&self) -> &[DebugPrimitive] {
        &self.primitives
    }

    /// Get all text entries.
    pub fn text_entries(&self) -> &[DebugTextEntry] {
        &self.text_entries
    }

    /// Get BT debug states.
    pub fn bt_states(&self) -> &[BtDebugState] {
        &self.bt_states
    }

    /// Get blackboard entries for an agent.
    pub fn blackboard(&self, agent_id: u32) -> Option<&[BlackboardDebugEntry]> {
        self.blackboard_entries.get(&agent_id).map(|v| v.as_slice())
    }

    /// Number of primitives queued.
    pub fn primitive_count(&self) -> usize {
        self.primitives.len()
    }
}

impl Default for AiDebugRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// An influence map cell for debug visualization.
#[derive(Debug, Clone)]
pub struct InfluenceCell {
    pub position: [f32; 3],
    pub size: f32,
    pub value: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_renderer() {
        let mut renderer = AiDebugRenderer::new();
        renderer.settings_mut().enabled = true;
        renderer.begin_frame();
        renderer.draw_awareness([0.0, 0.0, 0.0], AwarenessVisualLevel::Combat);
        assert!(renderer.primitive_count() > 0);
    }

    #[test]
    fn test_awareness_colors() {
        assert_eq!(AwarenessVisualLevel::Unaware.color().g, 1.0);
        assert_eq!(AwarenessVisualLevel::Combat.color().r, 1.0);
    }

    #[test]
    fn test_color_lerp() {
        let a = DebugColor::RED;
        let b = DebugColor::GREEN;
        let mid = a.lerp(&b, 0.5);
        assert!((mid.r - 0.5).abs() < 0.01);
        assert!((mid.g - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_nav_path() {
        let mut renderer = AiDebugRenderer::new();
        renderer.settings_mut().enabled = true;
        renderer.begin_frame();
        renderer.draw_nav_path(&[
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 5.0],
        ]);
        assert!(renderer.primitive_count() > 0);
    }
}
