//! Minimap system.
//!
//! Provides a top-down minimap with fog of war (explored/visible/hidden),
//! entity markers (friendly/enemy/objective/item), configurable zoom levels,
//! rotation modes (rotate-with-camera or north-up), minimap shapes
//! (circle/square), and custom icon support.
//!
//! # Key concepts
//!
//! - **MinimapConfig**: Global configuration for the minimap display.
//! - **FogOfWar**: Tracks which areas have been explored and which are
//!   currently visible (revealed by entities with vision).
//! - **MinimapMarker**: An icon/indicator on the minimap representing an
//!   entity, objective, or point of interest.
//! - **MinimapLayer**: Separate rendering layers (terrain, fog, markers,
//!   player indicator) composited into the final minimap image.
//! - **MinimapSystem**: Top-level manager that updates fog, markers, and
//!   provides render data to the UI.

use std::collections::HashMap;

use glam::Vec3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default minimap display size in UI pixels.
pub const DEFAULT_MINIMAP_SIZE: f32 = 200.0;

/// Default world radius visible on the minimap.
pub const DEFAULT_VIEW_RADIUS: f32 = 100.0;

/// Minimum zoom level (showing a large area).
pub const MIN_ZOOM: f32 = 0.25;

/// Maximum zoom level (showing a small area in detail).
pub const MAX_ZOOM: f32 = 4.0;

/// Default zoom level.
pub const DEFAULT_ZOOM: f32 = 1.0;

/// Zoom step when zooming in/out.
pub const ZOOM_STEP: f32 = 0.25;

/// Default fog of war grid resolution (cells per axis).
pub const DEFAULT_FOG_RESOLUTION: usize = 128;

/// Maximum fog of war grid resolution.
pub const MAX_FOG_RESOLUTION: usize = 512;

/// Maximum number of markers on the minimap.
pub const MAX_MARKERS: usize = 512;

/// Maximum number of custom icon types.
pub const MAX_ICON_TYPES: usize = 64;

/// Default vision radius for fog reveal (world units).
pub const DEFAULT_VISION_RADIUS: f32 = 20.0;

/// Alpha value for unexplored fog (fully hidden).
pub const FOG_HIDDEN_ALPHA: f32 = 1.0;

/// Alpha value for explored but not currently visible fog.
pub const FOG_EXPLORED_ALPHA: f32 = 0.5;

/// Alpha value for currently visible areas (no fog).
pub const FOG_VISIBLE_ALPHA: f32 = 0.0;

/// Marker pulse animation speed.
pub const MARKER_PULSE_SPEED: f32 = 2.0;

/// Maximum marker display size in UI pixels.
pub const MAX_MARKER_SIZE: f32 = 32.0;

/// Minimum marker display size in UI pixels.
pub const MIN_MARKER_SIZE: f32 = 4.0;

/// Default marker size.
pub const DEFAULT_MARKER_SIZE: f32 = 12.0;

/// Marker fade distance (fraction of view radius beyond which markers fade).
pub const MARKER_FADE_DISTANCE: f32 = 0.9;

// ---------------------------------------------------------------------------
// MinimapShape
// ---------------------------------------------------------------------------

/// Shape of the minimap display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MinimapShape {
    /// Circular minimap.
    Circle,
    /// Square/rectangular minimap.
    Square,
    /// Rounded rectangle.
    RoundedSquare,
}

// ---------------------------------------------------------------------------
// RotationMode
// ---------------------------------------------------------------------------

/// How the minimap rotates relative to the camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RotationMode {
    /// Minimap always shows north at the top.
    NorthUp,
    /// Minimap rotates so the player's facing direction is always up.
    RotateWithCamera,
    /// Player arrow rotates but map stays north-up.
    PlayerRotates,
}

// ---------------------------------------------------------------------------
// MarkerType
// ---------------------------------------------------------------------------

/// The type/category of a minimap marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkerType {
    /// The local player.
    Player,
    /// A friendly NPC or teammate.
    Friendly,
    /// An enemy entity.
    Enemy,
    /// A quest objective or waypoint.
    Objective,
    /// A collectible item or resource.
    Item,
    /// A shop or vendor.
    Shop,
    /// A quest giver NPC.
    QuestGiver,
    /// A fast travel point.
    FastTravel,
    /// A point of interest.
    PointOfInterest,
    /// A death location.
    Death,
    /// A custom marker (user-placed waypoint).
    Custom,
    /// A party member / group member.
    PartyMember,
    /// A boss enemy.
    Boss,
    /// Danger zone indicator.
    Danger,
}

impl MarkerType {
    /// Default color for this marker type (RGBA).
    pub fn default_color(&self) -> (u8, u8, u8, u8) {
        match self {
            Self::Player => (255, 255, 255, 255),
            Self::Friendly => (0, 200, 0, 255),
            Self::Enemy => (255, 0, 0, 255),
            Self::Objective => (255, 215, 0, 255),
            Self::Item => (0, 180, 255, 255),
            Self::Shop => (200, 150, 50, 255),
            Self::QuestGiver => (255, 255, 0, 255),
            Self::FastTravel => (100, 200, 255, 255),
            Self::PointOfInterest => (180, 180, 180, 255),
            Self::Death => (128, 0, 0, 255),
            Self::Custom => (255, 128, 0, 255),
            Self::PartyMember => (0, 255, 100, 255),
            Self::Boss => (255, 0, 50, 255),
            Self::Danger => (255, 50, 50, 200),
        }
    }

    /// Default size for this marker type.
    pub fn default_size(&self) -> f32 {
        match self {
            Self::Player => 16.0,
            Self::Boss => 20.0,
            Self::Objective => 18.0,
            Self::Danger => 24.0,
            _ => DEFAULT_MARKER_SIZE,
        }
    }

    /// Whether this marker type should pulse/animate.
    pub fn should_pulse(&self) -> bool {
        matches!(
            self,
            Self::Objective | Self::QuestGiver | Self::Boss | Self::Danger
        )
    }

    /// Z-order for rendering (higher = on top).
    pub fn z_order(&self) -> u32 {
        match self {
            Self::Player => 100,
            Self::PartyMember => 90,
            Self::Boss => 85,
            Self::Enemy => 80,
            Self::Objective => 75,
            Self::QuestGiver => 70,
            Self::Danger => 65,
            Self::Friendly => 60,
            Self::Shop => 50,
            Self::FastTravel => 45,
            Self::Item => 40,
            Self::PointOfInterest => 30,
            Self::Death => 20,
            Self::Custom => 10,
        }
    }
}

// ---------------------------------------------------------------------------
// MinimapMarker
// ---------------------------------------------------------------------------

/// An icon on the minimap representing an entity or point.
#[derive(Debug, Clone)]
pub struct MinimapMarker {
    /// Unique marker ID.
    pub id: u64,
    /// Entity ID this marker is attached to (0 = static).
    pub entity_id: u64,
    /// Marker type.
    pub marker_type: MarkerType,
    /// World position.
    pub world_position: Vec3,
    /// Facing direction (for arrow markers).
    pub facing: f32,
    /// Color override (None = use default).
    pub color: Option<(u8, u8, u8, u8)>,
    /// Size override (None = use default).
    pub size: Option<f32>,
    /// Custom icon ID (for custom icons).
    pub icon_id: Option<u32>,
    /// Display label.
    pub label: Option<String>,
    /// Whether the marker is visible.
    pub visible: bool,
    /// Whether the marker should show when off-screen (edge indicators).
    pub show_offscreen: bool,
    /// Whether the marker should be visible through fog.
    pub visible_through_fog: bool,
    /// Lifetime remaining (seconds, 0 = permanent).
    pub lifetime: f32,
    /// Whether the marker is pulsing.
    pub pulsing: bool,
    /// Current pulse phase.
    pulse_phase: f32,
    /// Whether this marker tracks an entity's position dynamically.
    pub tracking: bool,
    /// Tooltip text.
    pub tooltip: Option<String>,
    /// Distance from player (computed each frame).
    pub distance_to_player: f32,
}

impl MinimapMarker {
    /// Create a new marker.
    pub fn new(id: u64, marker_type: MarkerType, position: Vec3) -> Self {
        Self {
            id,
            entity_id: 0,
            marker_type,
            world_position: position,
            facing: 0.0,
            color: None,
            size: None,
            icon_id: None,
            label: None,
            visible: true,
            show_offscreen: false,
            visible_through_fog: false,
            lifetime: 0.0,
            pulsing: marker_type.should_pulse(),
            pulse_phase: 0.0,
            tracking: false,
            tooltip: None,
            distance_to_player: 0.0,
        }
    }

    /// Create a marker tracking an entity.
    pub fn tracking(id: u64, entity_id: u64, marker_type: MarkerType, position: Vec3) -> Self {
        let mut marker = Self::new(id, marker_type, position);
        marker.entity_id = entity_id;
        marker.tracking = true;
        marker
    }

    /// Set a label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set a custom color.
    pub fn with_color(mut self, r: u8, g: u8, b: u8, a: u8) -> Self {
        self.color = Some((r, g, b, a));
        self
    }

    /// Set a custom size.
    pub fn with_size(mut self, size: f32) -> Self {
        self.size = Some(size.clamp(MIN_MARKER_SIZE, MAX_MARKER_SIZE));
        self
    }

    /// Set a lifetime.
    pub fn with_lifetime(mut self, seconds: f32) -> Self {
        self.lifetime = seconds;
        self
    }

    /// Mark as visible through fog.
    pub fn through_fog(mut self) -> Self {
        self.visible_through_fog = true;
        self
    }

    /// Mark as showing off-screen.
    pub fn show_offscreen(mut self) -> Self {
        self.show_offscreen = true;
        self
    }

    /// Set tooltip text.
    pub fn with_tooltip(mut self, text: impl Into<String>) -> Self {
        self.tooltip = Some(text.into());
        self
    }

    /// Get the effective color.
    pub fn effective_color(&self) -> (u8, u8, u8, u8) {
        self.color.unwrap_or_else(|| self.marker_type.default_color())
    }

    /// Get the effective size.
    pub fn effective_size(&self) -> f32 {
        self.size.unwrap_or_else(|| self.marker_type.default_size())
    }

    /// Update pulse animation.
    pub fn update_pulse(&mut self, dt: f32) {
        if self.pulsing {
            self.pulse_phase += dt * MARKER_PULSE_SPEED;
            if self.pulse_phase > std::f32::consts::TAU {
                self.pulse_phase -= std::f32::consts::TAU;
            }
        }
    }

    /// Get the current pulse scale (for animation).
    pub fn pulse_scale(&self) -> f32 {
        if self.pulsing {
            1.0 + 0.2 * self.pulse_phase.sin()
        } else {
            1.0
        }
    }

    /// Update lifetime. Returns false if expired.
    pub fn update_lifetime(&mut self, dt: f32) -> bool {
        if self.lifetime > 0.0 {
            self.lifetime -= dt;
            self.lifetime > 0.0
        } else {
            true // permanent
        }
    }
}

// ---------------------------------------------------------------------------
// FogState
// ---------------------------------------------------------------------------

/// Fog of war state for a single cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FogState {
    /// Never seen — fully hidden.
    Hidden,
    /// Previously seen but not currently visible.
    Explored,
    /// Currently visible (within a revealer's range).
    Visible,
}

impl FogState {
    /// Get the fog alpha value.
    pub fn alpha(&self) -> f32 {
        match self {
            Self::Hidden => FOG_HIDDEN_ALPHA,
            Self::Explored => FOG_EXPLORED_ALPHA,
            Self::Visible => FOG_VISIBLE_ALPHA,
        }
    }
}

// ---------------------------------------------------------------------------
// FogOfWar
// ---------------------------------------------------------------------------

/// Grid-based fog of war system.
pub struct FogOfWar {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// Cell size in world units.
    pub cell_size: f32,
    /// World origin (bottom-left).
    pub origin: Vec3,
    /// Fog states for each cell.
    cells: Vec<FogState>,
    /// Visibility counters (how many revealers can see each cell).
    visibility_count: Vec<u32>,
    /// Whether fog was updated this frame.
    dirty: bool,
}

impl FogOfWar {
    /// Create a new fog of war grid.
    pub fn new(width: usize, height: usize, cell_size: f32, origin: Vec3) -> Self {
        let w = width.min(MAX_FOG_RESOLUTION);
        let h = height.min(MAX_FOG_RESOLUTION);
        let count = w * h;

        Self {
            width: w,
            height: h,
            cell_size,
            origin,
            cells: vec![FogState::Hidden; count],
            visibility_count: vec![0; count],
            dirty: true,
        }
    }

    /// Get the fog state at a grid position.
    pub fn get_state(&self, x: usize, z: usize) -> FogState {
        if x < self.width && z < self.height {
            self.cells[z * self.width + x]
        } else {
            FogState::Hidden
        }
    }

    /// Get the fog state at a world position.
    pub fn get_state_at_world(&self, pos: Vec3) -> FogState {
        if let Some((x, z)) = self.world_to_grid(pos) {
            self.get_state(x, z)
        } else {
            FogState::Hidden
        }
    }

    /// Check if a world position is visible.
    pub fn is_visible(&self, pos: Vec3) -> bool {
        self.get_state_at_world(pos) == FogState::Visible
    }

    /// Check if a world position has been explored.
    pub fn is_explored(&self, pos: Vec3) -> bool {
        let state = self.get_state_at_world(pos);
        state == FogState::Explored || state == FogState::Visible
    }

    /// Convert a world position to grid coordinates.
    pub fn world_to_grid(&self, pos: Vec3) -> Option<(usize, usize)> {
        let local = pos - self.origin;
        let x = (local.x / self.cell_size) as i32;
        let z = (local.z / self.cell_size) as i32;

        if x >= 0 && x < self.width as i32 && z >= 0 && z < self.height as i32 {
            Some((x as usize, z as usize))
        } else {
            None
        }
    }

    /// Convert grid coordinates to a world position.
    pub fn grid_to_world(&self, x: usize, z: usize) -> Vec3 {
        Vec3::new(
            self.origin.x + x as f32 * self.cell_size + self.cell_size * 0.5,
            self.origin.y,
            self.origin.z + z as f32 * self.cell_size + self.cell_size * 0.5,
        )
    }

    /// Begin a new frame: reset visibility counters (explored stays explored).
    pub fn begin_frame(&mut self) {
        for count in &mut self.visibility_count {
            *count = 0;
        }
        // Transition Visible -> Explored
        for cell in &mut self.cells {
            if *cell == FogState::Visible {
                *cell = FogState::Explored;
            }
        }
        self.dirty = true;
    }

    /// Reveal fog around a position with a given radius.
    pub fn reveal(&mut self, center: Vec3, radius: f32) {
        let cell_radius = (radius / self.cell_size).ceil() as i32;

        if let Some((cx, cz)) = self.world_to_grid(center) {
            let cx = cx as i32;
            let cz = cz as i32;

            for dz in -cell_radius..=cell_radius {
                for dx in -cell_radius..=cell_radius {
                    let gx = cx + dx;
                    let gz = cz + dz;

                    if gx < 0 || gx >= self.width as i32 || gz < 0 || gz >= self.height as i32 {
                        continue;
                    }

                    // Circle check
                    let dist_sq = (dx as f32).powi(2) + (dz as f32).powi(2);
                    let radius_cells = radius / self.cell_size;
                    if dist_sq > radius_cells * radius_cells {
                        continue;
                    }

                    let idx = gz as usize * self.width + gx as usize;
                    self.visibility_count[idx] += 1;
                    self.cells[idx] = FogState::Visible;
                }
            }
        }

        self.dirty = true;
    }

    /// Reveal the entire map.
    pub fn reveal_all(&mut self) {
        for cell in &mut self.cells {
            *cell = FogState::Visible;
        }
        self.dirty = true;
    }

    /// Hide the entire map (reset exploration).
    pub fn hide_all(&mut self) {
        for cell in &mut self.cells {
            *cell = FogState::Hidden;
        }
        for count in &mut self.visibility_count {
            *count = 0;
        }
        self.dirty = true;
    }

    /// Get the exploration percentage (0..1).
    pub fn exploration_percentage(&self) -> f32 {
        let explored = self
            .cells
            .iter()
            .filter(|c| **c != FogState::Hidden)
            .count();
        explored as f32 / self.cells.len() as f32
    }

    /// Get the raw fog data for rendering.
    pub fn fog_data(&self) -> &[FogState] {
        &self.cells
    }

    /// Whether the fog was updated this frame.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear the dirty flag.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }
}

// ---------------------------------------------------------------------------
// MinimapConfig
// ---------------------------------------------------------------------------

/// Configuration for the minimap display.
#[derive(Debug, Clone)]
pub struct MinimapConfig {
    /// Display size in UI pixels.
    pub display_size: f32,
    /// Display shape.
    pub shape: MinimapShape,
    /// Rotation mode.
    pub rotation_mode: RotationMode,
    /// Current zoom level.
    pub zoom: f32,
    /// World radius visible at zoom 1.0.
    pub base_view_radius: f32,
    /// Whether fog of war is enabled.
    pub fog_enabled: bool,
    /// Whether to show marker labels.
    pub show_labels: bool,
    /// Whether to show marker tooltips.
    pub show_tooltips: bool,
    /// UI position (x, y) from top-right corner.
    pub ui_offset: (f32, f32),
    /// Border color (RGBA).
    pub border_color: (u8, u8, u8, u8),
    /// Border width in pixels.
    pub border_width: f32,
    /// Background color (RGBA).
    pub background_color: (u8, u8, u8, u8),
    /// Whether the minimap is visible.
    pub visible: bool,
    /// Opacity (0..1).
    pub opacity: f32,
    /// Whether to show grid lines.
    pub show_grid: bool,
    /// Whether to show compass directions.
    pub show_compass: bool,
    /// Maximum markers shown at once (performance control).
    pub max_visible_markers: usize,
}

impl MinimapConfig {
    /// Create a default minimap configuration.
    pub fn new() -> Self {
        Self {
            display_size: DEFAULT_MINIMAP_SIZE,
            shape: MinimapShape::Circle,
            rotation_mode: RotationMode::RotateWithCamera,
            zoom: DEFAULT_ZOOM,
            base_view_radius: DEFAULT_VIEW_RADIUS,
            fog_enabled: true,
            show_labels: false,
            show_tooltips: true,
            ui_offset: (20.0, 20.0),
            border_color: (200, 200, 200, 255),
            border_width: 2.0,
            background_color: (20, 20, 30, 200),
            visible: true,
            opacity: 1.0,
            show_grid: false,
            show_compass: true,
            max_visible_markers: 128,
        }
    }

    /// Get the effective view radius.
    pub fn view_radius(&self) -> f32 {
        self.base_view_radius / self.zoom
    }

    /// Zoom in one step.
    pub fn zoom_in(&mut self) {
        self.zoom = (self.zoom + ZOOM_STEP).min(MAX_ZOOM);
    }

    /// Zoom out one step.
    pub fn zoom_out(&mut self) {
        self.zoom = (self.zoom - ZOOM_STEP).max(MIN_ZOOM);
    }

    /// Set zoom level directly.
    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
    }

    /// Convert world offset to minimap pixel coordinates.
    pub fn world_to_minimap(&self, offset: Vec3, camera_yaw: f32) -> (f32, f32) {
        let view_r = self.view_radius();
        let half_size = self.display_size * 0.5;

        let (dx, dz) = match self.rotation_mode {
            RotationMode::NorthUp | RotationMode::PlayerRotates => {
                (offset.x, offset.z)
            }
            RotationMode::RotateWithCamera => {
                let cos = camera_yaw.cos();
                let sin = camera_yaw.sin();
                (
                    offset.x * cos - offset.z * sin,
                    offset.x * sin + offset.z * cos,
                )
            }
        };

        let px = half_size + (dx / view_r) * half_size;
        let py = half_size - (dz / view_r) * half_size;

        (px, py)
    }

    /// Check if a minimap coordinate is within the display area.
    pub fn is_in_bounds(&self, x: f32, y: f32) -> bool {
        let half = self.display_size * 0.5;
        match self.shape {
            MinimapShape::Circle => {
                let dx = x - half;
                let dy = y - half;
                (dx * dx + dy * dy).sqrt() <= half
            }
            MinimapShape::Square | MinimapShape::RoundedSquare => {
                x >= 0.0 && x <= self.display_size && y >= 0.0 && y <= self.display_size
            }
        }
    }

    /// Cycle to the next rotation mode.
    pub fn cycle_rotation_mode(&mut self) {
        self.rotation_mode = match self.rotation_mode {
            RotationMode::NorthUp => RotationMode::RotateWithCamera,
            RotationMode::RotateWithCamera => RotationMode::PlayerRotates,
            RotationMode::PlayerRotates => RotationMode::NorthUp,
        };
    }

    /// Cycle to the next shape.
    pub fn cycle_shape(&mut self) {
        self.shape = match self.shape {
            MinimapShape::Circle => MinimapShape::Square,
            MinimapShape::Square => MinimapShape::RoundedSquare,
            MinimapShape::RoundedSquare => MinimapShape::Circle,
        };
    }
}

impl Default for MinimapConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CustomIcon
// ---------------------------------------------------------------------------

/// A custom icon definition for the minimap.
#[derive(Debug, Clone)]
pub struct CustomIcon {
    /// Unique icon ID.
    pub id: u32,
    /// Icon name.
    pub name: String,
    /// Texture/sprite path.
    pub texture: String,
    /// Default size in pixels.
    pub default_size: f32,
    /// Default color tint.
    pub tint: (u8, u8, u8, u8),
}

impl CustomIcon {
    /// Create a new custom icon.
    pub fn new(id: u32, name: impl Into<String>, texture: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            texture: texture.into(),
            default_size: DEFAULT_MARKER_SIZE,
            tint: (255, 255, 255, 255),
        }
    }
}

// ---------------------------------------------------------------------------
// MinimapRenderData
// ---------------------------------------------------------------------------

/// Data prepared for rendering the minimap each frame.
#[derive(Debug)]
pub struct MinimapRenderData {
    /// Player position in world space.
    pub player_position: Vec3,
    /// Player facing direction (yaw in radians).
    pub player_yaw: f32,
    /// Camera yaw (may differ from player yaw).
    pub camera_yaw: f32,
    /// Visible markers with computed screen positions.
    pub visible_markers: Vec<MarkerRenderData>,
    /// View radius in world units.
    pub view_radius: f32,
    /// Minimap display size.
    pub display_size: f32,
    /// Whether fog data was updated.
    pub fog_dirty: bool,
}

/// Per-marker render data.
#[derive(Debug)]
pub struct MarkerRenderData {
    /// Marker ID.
    pub marker_id: u64,
    /// Screen position on the minimap (pixels).
    pub screen_x: f32,
    pub screen_y: f32,
    /// Display size (pixels).
    pub size: f32,
    /// Color (RGBA).
    pub color: (u8, u8, u8, u8),
    /// Rotation angle.
    pub rotation: f32,
    /// Custom icon ID.
    pub icon_id: Option<u32>,
    /// Label text.
    pub label: Option<String>,
    /// Whether this marker is at the edge (off-screen indicator).
    pub clamped_to_edge: bool,
    /// Opacity.
    pub opacity: f32,
    /// Z-order.
    pub z_order: u32,
}

// ---------------------------------------------------------------------------
// MinimapSystem
// ---------------------------------------------------------------------------

/// Top-level minimap manager.
pub struct MinimapSystem {
    /// Configuration.
    pub config: MinimapConfig,
    /// Fog of war.
    pub fog: FogOfWar,
    /// All registered markers.
    markers: HashMap<u64, MinimapMarker>,
    /// Custom icons.
    custom_icons: HashMap<u32, CustomIcon>,
    /// Next marker ID.
    next_marker_id: u64,
    /// Player entity ID.
    pub player_entity: u64,
    /// Vision providers: (entity_id, radius).
    vision_providers: Vec<(u64, f32)>,
}

impl MinimapSystem {
    /// Create a new minimap system.
    pub fn new(
        world_width: f32,
        world_height: f32,
        fog_resolution: usize,
    ) -> Self {
        let res = fog_resolution.min(MAX_FOG_RESOLUTION);
        let cell_size = world_width / res as f32;
        let fog_height = (world_height / cell_size).ceil() as usize;

        Self {
            config: MinimapConfig::new(),
            fog: FogOfWar::new(res, fog_height.min(MAX_FOG_RESOLUTION), cell_size, Vec3::ZERO),
            markers: HashMap::new(),
            custom_icons: HashMap::new(),
            next_marker_id: 1,
            player_entity: 0,
            vision_providers: Vec::new(),
        }
    }

    /// Set the player entity.
    pub fn set_player(&mut self, entity_id: u64) {
        self.player_entity = entity_id;
    }

    // -----------------------------------------------------------------------
    // Markers
    // -----------------------------------------------------------------------

    /// Add a static marker. Returns the marker ID.
    pub fn add_marker(
        &mut self,
        marker_type: MarkerType,
        position: Vec3,
    ) -> u64 {
        if self.markers.len() >= MAX_MARKERS {
            return 0;
        }
        let id = self.next_marker_id;
        self.next_marker_id += 1;
        let marker = MinimapMarker::new(id, marker_type, position);
        self.markers.insert(id, marker);
        id
    }

    /// Add a tracking marker for an entity.
    pub fn add_entity_marker(
        &mut self,
        entity_id: u64,
        marker_type: MarkerType,
        position: Vec3,
    ) -> u64 {
        if self.markers.len() >= MAX_MARKERS {
            return 0;
        }
        let id = self.next_marker_id;
        self.next_marker_id += 1;
        let marker = MinimapMarker::tracking(id, entity_id, marker_type, position);
        self.markers.insert(id, marker);
        id
    }

    /// Remove a marker.
    pub fn remove_marker(&mut self, marker_id: u64) -> bool {
        self.markers.remove(&marker_id).is_some()
    }

    /// Remove all markers for an entity.
    pub fn remove_entity_markers(&mut self, entity_id: u64) {
        self.markers.retain(|_, m| m.entity_id != entity_id);
    }

    /// Get a marker by ID.
    pub fn get_marker(&self, marker_id: u64) -> Option<&MinimapMarker> {
        self.markers.get(&marker_id)
    }

    /// Get a mutable marker by ID.
    pub fn get_marker_mut(&mut self, marker_id: u64) -> Option<&mut MinimapMarker> {
        self.markers.get_mut(&marker_id)
    }

    /// Update a marker's position (for tracking markers).
    pub fn update_marker_position(&mut self, entity_id: u64, position: Vec3) {
        for marker in self.markers.values_mut() {
            if marker.tracking && marker.entity_id == entity_id {
                marker.world_position = position;
            }
        }
    }

    /// Update a marker's facing direction.
    pub fn update_marker_facing(&mut self, entity_id: u64, facing: f32) {
        for marker in self.markers.values_mut() {
            if marker.tracking && marker.entity_id == entity_id {
                marker.facing = facing;
            }
        }
    }

    /// Get marker count.
    pub fn marker_count(&self) -> usize {
        self.markers.len()
    }

    // -----------------------------------------------------------------------
    // Vision
    // -----------------------------------------------------------------------

    /// Register a vision provider (entity that reveals fog).
    pub fn add_vision_provider(&mut self, entity_id: u64, radius: f32) {
        self.vision_providers.push((entity_id, radius));
    }

    /// Remove a vision provider.
    pub fn remove_vision_provider(&mut self, entity_id: u64) {
        self.vision_providers.retain(|(e, _)| *e != entity_id);
    }

    // -----------------------------------------------------------------------
    // Custom icons
    // -----------------------------------------------------------------------

    /// Register a custom icon.
    pub fn register_icon(&mut self, icon: CustomIcon) {
        if self.custom_icons.len() < MAX_ICON_TYPES {
            self.custom_icons.insert(icon.id, icon);
        }
    }

    /// Get a custom icon.
    pub fn get_icon(&self, id: u32) -> Option<&CustomIcon> {
        self.custom_icons.get(&id)
    }

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /// Update the minimap each frame.
    pub fn update(
        &mut self,
        dt: f32,
        entity_positions: &HashMap<u64, Vec3>,
    ) {
        // Update fog
        if self.config.fog_enabled {
            self.fog.begin_frame();

            for &(entity_id, radius) in &self.vision_providers {
                if let Some(&pos) = entity_positions.get(&entity_id) {
                    self.fog.reveal(pos, radius);
                }
            }
        }

        // Update tracking markers
        for marker in self.markers.values_mut() {
            if marker.tracking {
                if let Some(&pos) = entity_positions.get(&marker.entity_id) {
                    marker.world_position = pos;
                }
            }

            // Update pulse animation
            marker.update_pulse(dt);

            // Update lifetime
            if !marker.update_lifetime(dt) {
                marker.visible = false;
            }
        }

        // Remove expired markers
        self.markers.retain(|_, m| m.visible || m.lifetime <= 0.0);

        // Calculate distances from player
        if let Some(&player_pos) = entity_positions.get(&self.player_entity) {
            for marker in self.markers.values_mut() {
                marker.distance_to_player = player_pos.distance(marker.world_position);
            }
        }
    }

    /// Generate render data for the current frame.
    pub fn render_data(
        &self,
        player_pos: Vec3,
        player_yaw: f32,
        camera_yaw: f32,
    ) -> MinimapRenderData {
        let view_radius = self.config.view_radius();
        let mut visible_markers = Vec::new();

        let mut sorted_markers: Vec<&MinimapMarker> = self
            .markers
            .values()
            .filter(|m| m.visible)
            .collect();

        // Sort by z-order
        sorted_markers.sort_by_key(|m| m.marker_type.z_order());

        for marker in sorted_markers.iter().take(self.config.max_visible_markers) {
            let offset = marker.world_position - player_pos;
            let distance = (offset.x * offset.x + offset.z * offset.z).sqrt();

            // Check fog visibility
            if self.config.fog_enabled && !marker.visible_through_fog {
                if !self.fog.is_visible(marker.world_position) {
                    continue;
                }
            }

            let (mut sx, mut sy) = self.config.world_to_minimap(offset, camera_yaw);

            let mut clamped = false;
            let in_bounds = self.config.is_in_bounds(sx, sy);

            if !in_bounds {
                if marker.show_offscreen {
                    // Clamp to edge
                    let half = self.config.display_size * 0.5;
                    let dx = sx - half;
                    let dy = sy - half;
                    let angle = dy.atan2(dx);
                    let edge = half - 5.0;
                    sx = half + angle.cos() * edge;
                    sy = half + angle.sin() * edge;
                    clamped = true;
                } else {
                    continue;
                }
            }

            // Opacity based on distance
            let opacity = if distance > view_radius * MARKER_FADE_DISTANCE {
                let fade = 1.0
                    - (distance - view_radius * MARKER_FADE_DISTANCE)
                        / (view_radius * (1.0 - MARKER_FADE_DISTANCE));
                fade.clamp(0.0, 1.0)
            } else {
                1.0
            };

            let size = marker.effective_size() * marker.pulse_scale();

            visible_markers.push(MarkerRenderData {
                marker_id: marker.id,
                screen_x: sx,
                screen_y: sy,
                size,
                color: marker.effective_color(),
                rotation: marker.facing,
                icon_id: marker.icon_id,
                label: marker.label.clone(),
                clamped_to_edge: clamped,
                opacity,
                z_order: marker.marker_type.z_order(),
            });
        }

        MinimapRenderData {
            player_position: player_pos,
            player_yaw,
            camera_yaw,
            visible_markers,
            view_radius,
            display_size: self.config.display_size,
            fog_dirty: self.fog.is_dirty(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fog_of_war() {
        let mut fog = FogOfWar::new(32, 32, 10.0, Vec3::ZERO);
        assert_eq!(fog.get_state(5, 5), FogState::Hidden);

        fog.reveal(Vec3::new(50.0, 0.0, 50.0), 30.0);
        assert_eq!(fog.get_state(5, 5), FogState::Visible);

        fog.begin_frame();
        assert_eq!(fog.get_state(5, 5), FogState::Explored);
    }

    #[test]
    fn test_minimap_config() {
        let mut config = MinimapConfig::new();
        assert_eq!(config.zoom, DEFAULT_ZOOM);

        config.zoom_in();
        assert!(config.zoom > DEFAULT_ZOOM);

        config.zoom_out();
        config.zoom_out();
        assert!(config.zoom < DEFAULT_ZOOM);
    }

    #[test]
    fn test_marker_creation() {
        let mut system = MinimapSystem::new(500.0, 500.0, 64);
        let id = system.add_marker(MarkerType::Objective, Vec3::new(100.0, 0.0, 100.0));
        assert!(id > 0);

        let marker = system.get_marker(id).unwrap();
        assert_eq!(marker.marker_type, MarkerType::Objective);
        assert!(marker.pulsing);
    }

    #[test]
    fn test_exploration_percentage() {
        let mut fog = FogOfWar::new(10, 10, 1.0, Vec3::ZERO);
        assert_eq!(fog.exploration_percentage(), 0.0);

        fog.reveal_all();
        assert_eq!(fog.exploration_percentage(), 1.0);
    }

    #[test]
    fn test_world_to_minimap() {
        let config = MinimapConfig::new();
        let offset = Vec3::new(0.0, 0.0, 0.0);
        let (x, y) = config.world_to_minimap(offset, 0.0);
        // Should be at center
        let half = config.display_size * 0.5;
        assert!((x - half).abs() < 0.01);
        assert!((y - half).abs() < 0.01);
    }
}
