//! Collision filtering system using layers and a collision matrix.
//!
//! Provides:
//! - **32 collision layers**: bitmask-based layer assignment for bodies
//! - **Layer collision matrix**: configurable NxN matrix for which layers collide
//! - **Per-body layer assignment**: each body can belong to one or more layers
//! - **Raycast layer filtering**: specify which layers a ray should hit
//! - **Trigger layer filtering**: separate filtering for trigger/sensor shapes
//! - **Layer groups**: named groups of layers for convenience
//! - **Presets**: common presets (Default, Player, Enemy, Environment, etc.)
//!
//! # Design
//!
//! Layers are represented as bits in a `u32` bitmask. The collision matrix is a
//! 32x32 symmetric matrix stored as an array of `u32` masks. Two bodies can
//! collide only if their layers overlap according to the matrix.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of collision layers (bits in u32).
pub const MAX_LAYERS: usize = 32;
/// Default layer (layer 0).
pub const DEFAULT_LAYER: u32 = 0;
/// All layers mask.
pub const ALL_LAYERS: u32 = 0xFFFFFFFF;
/// No layers mask.
pub const NO_LAYERS: u32 = 0;

// ---------------------------------------------------------------------------
// Layer bit helpers
// ---------------------------------------------------------------------------

/// Convert a layer index (0..31) to a bitmask.
#[inline]
pub fn layer_bit(layer: u32) -> u32 {
    debug_assert!(layer < MAX_LAYERS as u32, "Layer index out of range: {}", layer);
    1u32 << layer
}

/// Check if a bitmask contains a specific layer.
#[inline]
pub fn has_layer(mask: u32, layer: u32) -> bool {
    mask & layer_bit(layer) != 0
}

/// Combine multiple layer indices into a single mask.
pub fn layers_to_mask(layers: &[u32]) -> u32 {
    let mut mask = 0u32;
    for &layer in layers {
        if layer < MAX_LAYERS as u32 {
            mask |= 1u32 << layer;
        }
    }
    mask
}

/// Extract layer indices from a bitmask.
pub fn mask_to_layers(mask: u32) -> Vec<u32> {
    let mut layers = Vec::new();
    for i in 0..MAX_LAYERS as u32 {
        if mask & (1u32 << i) != 0 {
            layers.push(i);
        }
    }
    layers
}

/// Count the number of set layers in a mask.
#[inline]
pub fn layer_count(mask: u32) -> u32 {
    mask.count_ones()
}

// ---------------------------------------------------------------------------
// BuiltinLayer
// ---------------------------------------------------------------------------

/// Well-known built-in layers for common game use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum BuiltinLayer {
    /// Default layer for general objects.
    Default = 0,
    /// Player characters.
    Player = 1,
    /// Enemy characters.
    Enemy = 2,
    /// NPC characters (non-hostile).
    Npc = 3,
    /// Static environment (terrain, buildings).
    Environment = 4,
    /// Dynamic props (barrels, crates).
    Props = 5,
    /// Projectiles (bullets, arrows).
    Projectiles = 6,
    /// Trigger volumes (sensors, zones).
    Triggers = 7,
    /// Pickups and collectibles.
    Pickups = 8,
    /// Particle effects with collision.
    Particles = 9,
    /// Vehicles.
    Vehicles = 10,
    /// Destructible objects.
    Destructibles = 11,
    /// Water volumes.
    Water = 12,
    /// UI interaction raycasts.
    UI = 13,
    /// Ragdoll bodies.
    Ragdoll = 14,
    /// Debris from destruction.
    Debris = 15,
    /// Custom layer 0 (game-specific).
    Custom0 = 16,
    /// Custom layer 1.
    Custom1 = 17,
    /// Custom layer 2.
    Custom2 = 18,
    /// Custom layer 3.
    Custom3 = 19,
    /// Custom layer 4.
    Custom4 = 20,
    /// Custom layer 5.
    Custom5 = 21,
    /// Custom layer 6.
    Custom6 = 22,
    /// Custom layer 7.
    Custom7 = 23,
}

impl BuiltinLayer {
    /// Get the bitmask for this layer.
    #[inline]
    pub fn bit(&self) -> u32 {
        1u32 << (*self as u32)
    }

    /// Get the layer index.
    #[inline]
    pub fn index(&self) -> u32 {
        *self as u32
    }

    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            BuiltinLayer::Default => "Default",
            BuiltinLayer::Player => "Player",
            BuiltinLayer::Enemy => "Enemy",
            BuiltinLayer::Npc => "NPC",
            BuiltinLayer::Environment => "Environment",
            BuiltinLayer::Props => "Props",
            BuiltinLayer::Projectiles => "Projectiles",
            BuiltinLayer::Triggers => "Triggers",
            BuiltinLayer::Pickups => "Pickups",
            BuiltinLayer::Particles => "Particles",
            BuiltinLayer::Vehicles => "Vehicles",
            BuiltinLayer::Destructibles => "Destructibles",
            BuiltinLayer::Water => "Water",
            BuiltinLayer::UI => "UI",
            BuiltinLayer::Ragdoll => "Ragdoll",
            BuiltinLayer::Debris => "Debris",
            BuiltinLayer::Custom0 => "Custom0",
            BuiltinLayer::Custom1 => "Custom1",
            BuiltinLayer::Custom2 => "Custom2",
            BuiltinLayer::Custom3 => "Custom3",
            BuiltinLayer::Custom4 => "Custom4",
            BuiltinLayer::Custom5 => "Custom5",
            BuiltinLayer::Custom6 => "Custom6",
            BuiltinLayer::Custom7 => "Custom7",
        }
    }
}

// ---------------------------------------------------------------------------
// CollisionMatrix
// ---------------------------------------------------------------------------

/// 32x32 collision layer matrix determining which layers interact.
///
/// The matrix is symmetric: if layer A collides with layer B, then layer B
/// collides with layer A. Each row is a `u32` bitmask where bit N means
/// "this layer collides with layer N."
#[derive(Debug, Clone)]
pub struct CollisionMatrix {
    /// Row masks: `matrix[i]` is the set of layers that layer `i` collides with.
    matrix: [u32; MAX_LAYERS],
    /// Human-readable names for each layer.
    layer_names: [Option<String>; MAX_LAYERS],
}

impl CollisionMatrix {
    /// Create a new collision matrix where all layers collide with all layers.
    pub fn all_collide() -> Self {
        Self {
            matrix: [ALL_LAYERS; MAX_LAYERS],
            layer_names: Default::default(),
        }
    }

    /// Create a new collision matrix where no layers collide with each other.
    pub fn none_collide() -> Self {
        Self {
            matrix: [NO_LAYERS; MAX_LAYERS],
            layer_names: Default::default(),
        }
    }

    /// Create a default collision matrix with sensible game presets.
    pub fn default_game_preset() -> Self {
        let mut matrix = Self::none_collide();

        // Set up built-in layer names
        matrix.set_layer_name(BuiltinLayer::Default.index(), "Default");
        matrix.set_layer_name(BuiltinLayer::Player.index(), "Player");
        matrix.set_layer_name(BuiltinLayer::Enemy.index(), "Enemy");
        matrix.set_layer_name(BuiltinLayer::Npc.index(), "NPC");
        matrix.set_layer_name(BuiltinLayer::Environment.index(), "Environment");
        matrix.set_layer_name(BuiltinLayer::Props.index(), "Props");
        matrix.set_layer_name(BuiltinLayer::Projectiles.index(), "Projectiles");
        matrix.set_layer_name(BuiltinLayer::Triggers.index(), "Triggers");
        matrix.set_layer_name(BuiltinLayer::Pickups.index(), "Pickups");
        matrix.set_layer_name(BuiltinLayer::Vehicles.index(), "Vehicles");
        matrix.set_layer_name(BuiltinLayer::Destructibles.index(), "Destructibles");
        matrix.set_layer_name(BuiltinLayer::Water.index(), "Water");
        matrix.set_layer_name(BuiltinLayer::Ragdoll.index(), "Ragdoll");
        matrix.set_layer_name(BuiltinLayer::Debris.index(), "Debris");

        // Default collides with environment, props, destructibles
        matrix.set_collides(BuiltinLayer::Default.index(), BuiltinLayer::Environment.index(), true);
        matrix.set_collides(BuiltinLayer::Default.index(), BuiltinLayer::Props.index(), true);
        matrix.set_collides(BuiltinLayer::Default.index(), BuiltinLayer::Default.index(), true);

        // Player collides with environment, props, enemy, NPC, vehicles, pickups, water
        let player = BuiltinLayer::Player.index();
        matrix.set_collides(player, BuiltinLayer::Environment.index(), true);
        matrix.set_collides(player, BuiltinLayer::Props.index(), true);
        matrix.set_collides(player, BuiltinLayer::Enemy.index(), true);
        matrix.set_collides(player, BuiltinLayer::Npc.index(), true);
        matrix.set_collides(player, BuiltinLayer::Vehicles.index(), true);
        matrix.set_collides(player, BuiltinLayer::Pickups.index(), true);
        matrix.set_collides(player, BuiltinLayer::Water.index(), true);
        matrix.set_collides(player, BuiltinLayer::Destructibles.index(), true);

        // Enemy collides with environment, props, player, other enemies, vehicles
        let enemy = BuiltinLayer::Enemy.index();
        matrix.set_collides(enemy, BuiltinLayer::Environment.index(), true);
        matrix.set_collides(enemy, BuiltinLayer::Props.index(), true);
        matrix.set_collides(enemy, enemy, true);
        matrix.set_collides(enemy, BuiltinLayer::Vehicles.index(), true);
        matrix.set_collides(enemy, BuiltinLayer::Destructibles.index(), true);

        // Projectiles collide with environment, player, enemy, NPC, props, vehicles, destructibles
        let proj = BuiltinLayer::Projectiles.index();
        matrix.set_collides(proj, BuiltinLayer::Environment.index(), true);
        matrix.set_collides(proj, BuiltinLayer::Player.index(), true);
        matrix.set_collides(proj, BuiltinLayer::Enemy.index(), true);
        matrix.set_collides(proj, BuiltinLayer::Npc.index(), true);
        matrix.set_collides(proj, BuiltinLayer::Props.index(), true);
        matrix.set_collides(proj, BuiltinLayer::Vehicles.index(), true);
        matrix.set_collides(proj, BuiltinLayer::Destructibles.index(), true);

        // Vehicles collide with environment
        matrix.set_collides(BuiltinLayer::Vehicles.index(), BuiltinLayer::Environment.index(), true);
        matrix.set_collides(BuiltinLayer::Vehicles.index(), BuiltinLayer::Vehicles.index(), true);

        // Ragdoll collides with environment, props
        matrix.set_collides(BuiltinLayer::Ragdoll.index(), BuiltinLayer::Environment.index(), true);
        matrix.set_collides(BuiltinLayer::Ragdoll.index(), BuiltinLayer::Props.index(), true);
        matrix.set_collides(BuiltinLayer::Ragdoll.index(), BuiltinLayer::Ragdoll.index(), true);

        // Debris collides with environment only (to avoid performance issues)
        matrix.set_collides(BuiltinLayer::Debris.index(), BuiltinLayer::Environment.index(), true);

        // NPC collides with environment, props
        matrix.set_collides(BuiltinLayer::Npc.index(), BuiltinLayer::Environment.index(), true);
        matrix.set_collides(BuiltinLayer::Npc.index(), BuiltinLayer::Props.index(), true);
        matrix.set_collides(BuiltinLayer::Npc.index(), BuiltinLayer::Npc.index(), true);

        matrix
    }

    /// Set whether two layers collide (symmetric).
    pub fn set_collides(&mut self, layer_a: u32, layer_b: u32, collides: bool) {
        if layer_a >= MAX_LAYERS as u32 || layer_b >= MAX_LAYERS as u32 {
            return;
        }
        if collides {
            self.matrix[layer_a as usize] |= 1u32 << layer_b;
            self.matrix[layer_b as usize] |= 1u32 << layer_a;
        } else {
            self.matrix[layer_a as usize] &= !(1u32 << layer_b);
            self.matrix[layer_b as usize] &= !(1u32 << layer_a);
        }
    }

    /// Check if two layers can collide.
    #[inline]
    pub fn can_collide(&self, layer_a: u32, layer_b: u32) -> bool {
        if layer_a >= MAX_LAYERS as u32 || layer_b >= MAX_LAYERS as u32 {
            return false;
        }
        self.matrix[layer_a as usize] & (1u32 << layer_b) != 0
    }

    /// Check if two bitmasks can collide (any overlapping layer pair).
    #[inline]
    pub fn masks_can_collide(&self, mask_a: u32, mask_b: u32) -> bool {
        for layer_a in 0..MAX_LAYERS as u32 {
            if mask_a & (1u32 << layer_a) == 0 {
                continue;
            }
            if self.matrix[layer_a as usize] & mask_b != 0 {
                return true;
            }
        }
        false
    }

    /// Get the collision mask for a specific layer.
    #[inline]
    pub fn collision_mask(&self, layer: u32) -> u32 {
        if layer >= MAX_LAYERS as u32 {
            return NO_LAYERS;
        }
        self.matrix[layer as usize]
    }

    /// Get the combined collision mask for a bitmask of layers.
    pub fn combined_collision_mask(&self, layers_mask: u32) -> u32 {
        let mut result = 0u32;
        for layer in 0..MAX_LAYERS as u32 {
            if layers_mask & (1u32 << layer) != 0 {
                result |= self.matrix[layer as usize];
            }
        }
        result
    }

    /// Set a name for a layer.
    pub fn set_layer_name(&mut self, layer: u32, name: &str) {
        if layer < MAX_LAYERS as u32 {
            self.layer_names[layer as usize] = Some(name.to_string());
        }
    }

    /// Get the name of a layer.
    pub fn layer_name(&self, layer: u32) -> Option<&str> {
        if layer < MAX_LAYERS as u32 {
            self.layer_names[layer as usize].as_deref()
        } else {
            None
        }
    }

    /// Enable collision between a layer and all other layers.
    pub fn enable_all_collisions(&mut self, layer: u32) {
        if layer >= MAX_LAYERS as u32 {
            return;
        }
        self.matrix[layer as usize] = ALL_LAYERS;
        for i in 0..MAX_LAYERS {
            self.matrix[i] |= 1u32 << layer;
        }
    }

    /// Disable all collisions for a layer.
    pub fn disable_all_collisions(&mut self, layer: u32) {
        if layer >= MAX_LAYERS as u32 {
            return;
        }
        self.matrix[layer as usize] = NO_LAYERS;
        for i in 0..MAX_LAYERS {
            self.matrix[i] &= !(1u32 << layer);
        }
    }

    /// Get a debug string representation of the matrix.
    pub fn debug_string(&self) -> String {
        let mut s = String::new();
        s.push_str("Collision Matrix:\n");
        for i in 0..MAX_LAYERS {
            if self.matrix[i] == NO_LAYERS && self.layer_names[i].is_none() {
                continue;
            }
            let name = self.layer_names[i].as_deref().unwrap_or("unnamed");
            s.push_str(&format!("  Layer {:2} ({}): {:032b}\n", i, name, self.matrix[i]));
        }
        s
    }
}

impl Default for CollisionMatrix {
    fn default() -> Self {
        Self::default_game_preset()
    }
}

// ---------------------------------------------------------------------------
// LayerFilter
// ---------------------------------------------------------------------------

/// Filter for physics queries (raycasts, overlap tests, etc.).
#[derive(Debug, Clone)]
pub struct LayerFilter {
    /// Layers to include in the query (bitmask).
    pub include_layers: u32,
    /// Layers to exclude from the query (bitmask, takes precedence over include).
    pub exclude_layers: u32,
    /// Whether to use the collision matrix for additional filtering.
    pub use_collision_matrix: bool,
    /// Source layer for collision matrix lookups (used when `use_collision_matrix` is true).
    pub source_layer: u32,
}

impl LayerFilter {
    /// Create a filter that includes all layers.
    pub fn all() -> Self {
        Self {
            include_layers: ALL_LAYERS,
            exclude_layers: NO_LAYERS,
            use_collision_matrix: false,
            source_layer: 0,
        }
    }

    /// Create a filter that includes no layers.
    pub fn none() -> Self {
        Self {
            include_layers: NO_LAYERS,
            exclude_layers: NO_LAYERS,
            use_collision_matrix: false,
            source_layer: 0,
        }
    }

    /// Create a filter for specific layers.
    pub fn only(layers: u32) -> Self {
        Self {
            include_layers: layers,
            exclude_layers: NO_LAYERS,
            use_collision_matrix: false,
            source_layer: 0,
        }
    }

    /// Create a filter that excludes specific layers.
    pub fn except(layers: u32) -> Self {
        Self {
            include_layers: ALL_LAYERS,
            exclude_layers: layers,
            use_collision_matrix: false,
            source_layer: 0,
        }
    }

    /// Create a filter from a source layer using the collision matrix.
    pub fn from_collision_matrix(source_layer: u32) -> Self {
        Self {
            include_layers: ALL_LAYERS,
            exclude_layers: NO_LAYERS,
            use_collision_matrix: true,
            source_layer,
        }
    }

    /// Test if a body layer mask passes this filter.
    pub fn passes(&self, body_layers: u32, matrix: Option<&CollisionMatrix>) -> bool {
        // Check exclusion first
        if body_layers & self.exclude_layers != 0 {
            return false;
        }
        // Check inclusion
        if body_layers & self.include_layers == 0 {
            return false;
        }
        // Check collision matrix if requested
        if self.use_collision_matrix {
            if let Some(m) = matrix {
                return m.masks_can_collide(layer_bit(self.source_layer), body_layers);
            }
        }
        true
    }

    /// Builder: add include layers.
    pub fn with_include(mut self, layers: u32) -> Self {
        self.include_layers |= layers;
        self
    }

    /// Builder: add exclude layers.
    pub fn with_exclude(mut self, layers: u32) -> Self {
        self.exclude_layers |= layers;
        self
    }
}

impl Default for LayerFilter {
    fn default() -> Self {
        Self::all()
    }
}

// ---------------------------------------------------------------------------
// TriggerLayerFilter
// ---------------------------------------------------------------------------

/// Specialized filter for trigger/sensor volumes.
#[derive(Debug, Clone)]
pub struct TriggerLayerFilter {
    /// Layers that this trigger responds to.
    pub responsive_layers: u32,
    /// Layers that are ignored by this trigger.
    pub ignored_layers: u32,
    /// Whether the trigger only responds to its own layer.
    pub self_layer_only: bool,
    /// The trigger's own layer.
    pub trigger_layer: u32,
}

impl TriggerLayerFilter {
    /// Create a trigger filter that responds to all layers.
    pub fn all() -> Self {
        Self {
            responsive_layers: ALL_LAYERS,
            ignored_layers: NO_LAYERS,
            self_layer_only: false,
            trigger_layer: 0,
        }
    }

    /// Create a trigger filter for specific layers.
    pub fn for_layers(layers: u32) -> Self {
        Self {
            responsive_layers: layers,
            ignored_layers: NO_LAYERS,
            self_layer_only: false,
            trigger_layer: 0,
        }
    }

    /// Test if an entity's layers pass this trigger filter.
    pub fn should_trigger(&self, entity_layers: u32) -> bool {
        if self.self_layer_only {
            return entity_layers & layer_bit(self.trigger_layer) != 0;
        }
        if entity_layers & self.ignored_layers != 0 {
            return false;
        }
        entity_layers & self.responsive_layers != 0
    }
}

impl Default for TriggerLayerFilter {
    fn default() -> Self {
        Self::all()
    }
}

// ---------------------------------------------------------------------------
// LayerGroup
// ---------------------------------------------------------------------------

/// A named group of layers for convenience.
#[derive(Debug, Clone)]
pub struct LayerGroup {
    /// Name of the group.
    pub name: String,
    /// Bitmask of layers in this group.
    pub mask: u32,
    /// Description.
    pub description: String,
}

impl LayerGroup {
    /// Create a new layer group.
    pub fn new(name: impl Into<String>, mask: u32) -> Self {
        Self {
            name: name.into(),
            mask,
            description: String::new(),
        }
    }

    /// Builder: set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Check if a layer is in this group.
    pub fn contains_layer(&self, layer: u32) -> bool {
        has_layer(self.mask, layer)
    }

    /// Check if a mask overlaps with this group.
    pub fn overlaps(&self, mask: u32) -> bool {
        self.mask & mask != 0
    }
}

// ---------------------------------------------------------------------------
// LayerGroupManager
// ---------------------------------------------------------------------------

/// Manages named groups of collision layers.
#[derive(Debug)]
pub struct LayerGroupManager {
    /// Named groups.
    groups: HashMap<String, LayerGroup>,
}

impl LayerGroupManager {
    /// Create a new layer group manager.
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
        }
    }

    /// Create with default groups.
    pub fn with_defaults() -> Self {
        let mut mgr = Self::new();
        mgr.add_group(LayerGroup::new(
            "Characters",
            BuiltinLayer::Player.bit() | BuiltinLayer::Enemy.bit() | BuiltinLayer::Npc.bit(),
        ).with_description("All character entities"));

        mgr.add_group(LayerGroup::new(
            "Solids",
            BuiltinLayer::Environment.bit() | BuiltinLayer::Props.bit() | BuiltinLayer::Vehicles.bit(),
        ).with_description("Solid colliders"));

        mgr.add_group(LayerGroup::new(
            "Hittable",
            BuiltinLayer::Player.bit() | BuiltinLayer::Enemy.bit() | BuiltinLayer::Npc.bit()
                | BuiltinLayer::Props.bit() | BuiltinLayer::Vehicles.bit() | BuiltinLayer::Destructibles.bit(),
        ).with_description("Things that projectiles can hit"));

        mgr.add_group(LayerGroup::new(
            "NonInteractive",
            BuiltinLayer::Triggers.bit() | BuiltinLayer::Particles.bit() | BuiltinLayer::UI.bit(),
        ).with_description("Non-physical interactive layers"));

        mgr
    }

    /// Add a group.
    pub fn add_group(&mut self, group: LayerGroup) {
        self.groups.insert(group.name.clone(), group);
    }

    /// Remove a group by name.
    pub fn remove_group(&mut self, name: &str) -> Option<LayerGroup> {
        self.groups.remove(name)
    }

    /// Get a group by name.
    pub fn group(&self, name: &str) -> Option<&LayerGroup> {
        self.groups.get(name)
    }

    /// Get the mask for a group by name.
    pub fn group_mask(&self, name: &str) -> u32 {
        self.groups.get(name).map_or(NO_LAYERS, |g| g.mask)
    }

    /// List all group names.
    pub fn group_names(&self) -> Vec<&str> {
        self.groups.keys().map(|k| k.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// PhysicsLayerComponent (ECS)
// ---------------------------------------------------------------------------

/// ECS component assigning collision layers to an entity.
#[derive(Debug, Clone)]
pub struct PhysicsLayerComponent {
    /// The layers this body belongs to (bitmask).
    pub layers: u32,
    /// Override collision mask (if set, used instead of the matrix).
    pub collision_mask_override: Option<u32>,
    /// Whether this body is a trigger/sensor.
    pub is_trigger: bool,
    /// Trigger filter (used when `is_trigger` is true).
    pub trigger_filter: TriggerLayerFilter,
}

impl PhysicsLayerComponent {
    /// Create a component on the default layer.
    pub fn default_layer() -> Self {
        Self {
            layers: layer_bit(BuiltinLayer::Default.index()),
            collision_mask_override: None,
            is_trigger: false,
            trigger_filter: TriggerLayerFilter::default(),
        }
    }

    /// Create a component on a specific layer.
    pub fn on_layer(layer: BuiltinLayer) -> Self {
        Self {
            layers: layer.bit(),
            collision_mask_override: None,
            is_trigger: false,
            trigger_filter: TriggerLayerFilter::default(),
        }
    }

    /// Create a component on multiple layers.
    pub fn on_layers(layers: &[BuiltinLayer]) -> Self {
        let mut mask = 0u32;
        for layer in layers {
            mask |= layer.bit();
        }
        Self {
            layers: mask,
            collision_mask_override: None,
            is_trigger: false,
            trigger_filter: TriggerLayerFilter::default(),
        }
    }

    /// Create a trigger component.
    pub fn trigger(layer: BuiltinLayer, responsive_layers: u32) -> Self {
        Self {
            layers: layer.bit(),
            collision_mask_override: None,
            is_trigger: true,
            trigger_filter: TriggerLayerFilter::for_layers(responsive_layers),
        }
    }

    /// Add a layer to this component.
    pub fn add_layer(&mut self, layer: u32) {
        self.layers |= layer_bit(layer);
    }

    /// Remove a layer from this component.
    pub fn remove_layer(&mut self, layer: u32) {
        self.layers &= !layer_bit(layer);
    }

    /// Check if this component is on a specific layer.
    pub fn is_on_layer(&self, layer: u32) -> bool {
        has_layer(self.layers, layer)
    }

    /// Check if this body should collide with another body.
    pub fn should_collide_with(&self, other: &PhysicsLayerComponent, matrix: &CollisionMatrix) -> bool {
        if let Some(override_mask) = self.collision_mask_override {
            return other.layers & override_mask != 0;
        }
        matrix.masks_can_collide(self.layers, other.layers)
    }
}

impl Default for PhysicsLayerComponent {
    fn default() -> Self {
        Self::default_layer()
    }
}

// ---------------------------------------------------------------------------
// PhysicsLayerSystem
// ---------------------------------------------------------------------------

/// System that manages the collision matrix and layer groups.
pub struct PhysicsLayerSystem {
    /// The collision matrix.
    pub matrix: CollisionMatrix,
    /// Layer group manager.
    pub groups: LayerGroupManager,
    /// Statistics.
    collision_checks: u64,
    collision_passes: u64,
    collision_rejects: u64,
}

impl PhysicsLayerSystem {
    /// Create a new system with default game presets.
    pub fn new() -> Self {
        Self {
            matrix: CollisionMatrix::default_game_preset(),
            groups: LayerGroupManager::with_defaults(),
            collision_checks: 0,
            collision_passes: 0,
            collision_rejects: 0,
        }
    }

    /// Create with a custom collision matrix.
    pub fn with_matrix(matrix: CollisionMatrix) -> Self {
        Self {
            matrix,
            groups: LayerGroupManager::new(),
            collision_checks: 0,
            collision_passes: 0,
            collision_rejects: 0,
        }
    }

    /// Test if two bodies should collide.
    pub fn should_collide(&mut self, a: &PhysicsLayerComponent, b: &PhysicsLayerComponent) -> bool {
        self.collision_checks += 1;
        let result = a.should_collide_with(b, &self.matrix);
        if result {
            self.collision_passes += 1;
        } else {
            self.collision_rejects += 1;
        }
        result
    }

    /// Test if a raycast should hit a body.
    pub fn raycast_filter(&mut self, filter: &LayerFilter, body: &PhysicsLayerComponent) -> bool {
        self.collision_checks += 1;
        let result = filter.passes(body.layers, Some(&self.matrix));
        if result {
            self.collision_passes += 1;
        } else {
            self.collision_rejects += 1;
        }
        result
    }

    /// Test if a trigger should fire for an entity.
    pub fn trigger_filter(&self, trigger: &PhysicsLayerComponent, entity: &PhysicsLayerComponent) -> bool {
        if !trigger.is_trigger {
            return false;
        }
        trigger.trigger_filter.should_trigger(entity.layers)
    }

    /// Get statistics.
    pub fn stats(&self) -> PhysicsLayerStats {
        PhysicsLayerStats {
            total_checks: self.collision_checks,
            passes: self.collision_passes,
            rejects: self.collision_rejects,
            rejection_rate: if self.collision_checks > 0 {
                self.collision_rejects as f64 / self.collision_checks as f64
            } else {
                0.0
            },
        }
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.collision_checks = 0;
        self.collision_passes = 0;
        self.collision_rejects = 0;
    }
}

/// Statistics for the physics layer system.
#[derive(Debug, Clone)]
pub struct PhysicsLayerStats {
    /// Total collision checks performed.
    pub total_checks: u64,
    /// Checks that passed (collision possible).
    pub passes: u64,
    /// Checks that were rejected (collision filtered out).
    pub rejects: u64,
    /// Rejection rate (0..1).
    pub rejection_rate: f64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_bit() {
        assert_eq!(layer_bit(0), 1);
        assert_eq!(layer_bit(1), 2);
        assert_eq!(layer_bit(5), 32);
        assert_eq!(layer_bit(31), 0x80000000);
    }

    #[test]
    fn test_has_layer() {
        let mask = layer_bit(0) | layer_bit(3) | layer_bit(7);
        assert!(has_layer(mask, 0));
        assert!(has_layer(mask, 3));
        assert!(has_layer(mask, 7));
        assert!(!has_layer(mask, 1));
        assert!(!has_layer(mask, 4));
    }

    #[test]
    fn test_collision_matrix_symmetric() {
        let mut matrix = CollisionMatrix::none_collide();
        matrix.set_collides(1, 5, true);
        assert!(matrix.can_collide(1, 5));
        assert!(matrix.can_collide(5, 1));
        matrix.set_collides(1, 5, false);
        assert!(!matrix.can_collide(1, 5));
        assert!(!matrix.can_collide(5, 1));
    }

    #[test]
    fn test_default_preset_player_enemy() {
        let matrix = CollisionMatrix::default_game_preset();
        assert!(matrix.can_collide(BuiltinLayer::Player.index(), BuiltinLayer::Enemy.index()));
        assert!(matrix.can_collide(BuiltinLayer::Player.index(), BuiltinLayer::Environment.index()));
    }

    #[test]
    fn test_layer_filter() {
        let filter = LayerFilter::only(BuiltinLayer::Player.bit() | BuiltinLayer::Enemy.bit());
        assert!(filter.passes(BuiltinLayer::Player.bit(), None));
        assert!(filter.passes(BuiltinLayer::Enemy.bit(), None));
        assert!(!filter.passes(BuiltinLayer::Props.bit(), None));
    }

    #[test]
    fn test_layer_filter_exclude() {
        let filter = LayerFilter::except(BuiltinLayer::Debris.bit());
        assert!(filter.passes(BuiltinLayer::Player.bit(), None));
        assert!(!filter.passes(BuiltinLayer::Debris.bit(), None));
    }

    #[test]
    fn test_component_collision() {
        let matrix = CollisionMatrix::default_game_preset();
        let player = PhysicsLayerComponent::on_layer(BuiltinLayer::Player);
        let enemy = PhysicsLayerComponent::on_layer(BuiltinLayer::Enemy);
        let trigger = PhysicsLayerComponent::on_layer(BuiltinLayer::Triggers);

        assert!(player.should_collide_with(&enemy, &matrix));
        assert!(!player.should_collide_with(&trigger, &matrix));
    }

    #[test]
    fn test_layers_to_mask() {
        let mask = layers_to_mask(&[0, 3, 7]);
        assert_eq!(mask, 0b10001001);
    }

    #[test]
    fn test_mask_to_layers() {
        let layers = mask_to_layers(0b10001001);
        assert_eq!(layers, vec![0, 3, 7]);
    }
}
