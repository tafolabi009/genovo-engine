// =============================================================================
// Genovo Engine - Terrain Editing Tools
// =============================================================================
//
// Provides brush-based terrain sculpting, texture painting, vegetation
// density painting, erosion simulation, and heightmap stamp importing.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Brush Types
// ---------------------------------------------------------------------------

/// The type of terrain modification a brush performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BrushType {
    /// Raise the terrain height.
    Raise,
    /// Lower the terrain height.
    Lower,
    /// Smooth terrain towards the local average.
    Smooth,
    /// Flatten terrain to the brush-start height.
    Flatten,
    /// Add procedural noise to the terrain.
    Noise,
    /// Paint terrain texture/splatmap layer.
    PaintTexture,
    /// Paint vegetation density.
    PaintVegetation,
    /// Stamp a heightmap pattern.
    Stamp,
}

impl BrushType {
    /// Display-friendly name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Raise => "Raise",
            Self::Lower => "Lower",
            Self::Smooth => "Smooth",
            Self::Flatten => "Flatten",
            Self::Noise => "Noise",
            Self::PaintTexture => "Paint Texture",
            Self::PaintVegetation => "Paint Vegetation",
            Self::Stamp => "Stamp",
        }
    }

    /// All brush types.
    pub fn all() -> &'static [BrushType] {
        &[
            Self::Raise, Self::Lower, Self::Smooth, Self::Flatten,
            Self::Noise, Self::PaintTexture, Self::PaintVegetation, Self::Stamp,
        ]
    }
}

/// Falloff curve shape for brush attenuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BrushFalloff {
    /// Linear falloff from center to edge.
    Linear,
    /// Smooth (quadratic) falloff.
    Smooth,
    /// Constant strength across the entire brush.
    Constant,
    /// Sharp peak at center.
    Sharp,
    /// Custom falloff using a curve.
    Custom,
}

impl Default for BrushFalloff {
    fn default() -> Self {
        Self::Smooth
    }
}

// ---------------------------------------------------------------------------
// Brush Settings
// ---------------------------------------------------------------------------

/// Configuration for a terrain brush.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrushSettings {
    /// Brush radius in world units.
    pub size: f32,
    /// Brush strength / intensity [0.0, 1.0].
    pub strength: f32,
    /// Falloff curve.
    pub falloff: BrushFalloff,
    /// Falloff exponent (controls the curve shape for Smooth/Sharp).
    pub falloff_exponent: f32,
    /// Brush rotation in degrees.
    pub rotation: f32,
    /// Spacing between brush stamps along a stroke (as fraction of size).
    pub spacing: f32,
    /// Noise frequency (for Noise brush).
    pub noise_frequency: f32,
    /// Noise amplitude (for Noise brush).
    pub noise_amplitude: f32,
    /// Flatten target height (for Flatten brush, set when stroke begins).
    pub flatten_height: Option<f32>,
    /// Active splatmap layer index (for PaintTexture).
    pub paint_layer: u32,
    /// Vegetation type index (for PaintVegetation).
    pub vegetation_type: u32,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            size: 10.0,
            strength: 0.5,
            falloff: BrushFalloff::Smooth,
            falloff_exponent: 2.0,
            rotation: 0.0,
            spacing: 0.25,
            noise_frequency: 0.1,
            noise_amplitude: 1.0,
            flatten_height: None,
            paint_layer: 0,
            vegetation_type: 0,
        }
    }
}

impl BrushSettings {
    /// Compute the falloff value at a given distance from center.
    /// Returns a value in [0.0, 1.0].
    pub fn compute_falloff(&self, distance: f32) -> f32 {
        if distance >= self.size {
            return 0.0;
        }
        let t = distance / self.size.max(0.001);

        match self.falloff {
            BrushFalloff::Constant => 1.0,
            BrushFalloff::Linear => 1.0 - t,
            BrushFalloff::Smooth => {
                let s = 1.0 - t;
                s.powf(self.falloff_exponent)
            }
            BrushFalloff::Sharp => {
                let s = 1.0 - t;
                s.powf(self.falloff_exponent * 3.0)
            }
            BrushFalloff::Custom => 1.0 - t, // Default to linear for custom.
        }
    }

    /// Compute the effective delta per application of the brush at a point.
    pub fn effective_strength(&self, distance: f32, dt: f32) -> f32 {
        self.strength * self.compute_falloff(distance) * dt
    }
}

// ---------------------------------------------------------------------------
// Terrain Brush Stroke
// ---------------------------------------------------------------------------

/// A recorded brush stroke for undo/redo support.
#[derive(Debug, Clone)]
pub struct TerrainBrushStroke {
    /// Unique stroke identifier.
    pub id: Uuid,
    /// Brush type used for this stroke.
    pub brush_type: BrushType,
    /// Brush settings at the time of the stroke.
    pub settings: BrushSettings,
    /// Sequence of brush positions along the stroke path.
    pub points: Vec<StrokePoint>,
    /// Snapshot of modified heightmap data before the stroke (for undo).
    /// Key: (tile_x, tile_z), Value: (local_x, local_z, old_height).
    pub height_snapshot: Vec<HeightSample>,
    /// Snapshot of modified splatmap data before the stroke (for undo).
    pub splatmap_snapshot: Vec<SplatmapSample>,
    /// Whether this stroke has been completed (mouse released).
    pub completed: bool,
}

/// A single point along a brush stroke.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StrokePoint {
    /// World-space position of the brush center.
    pub position: [f32; 3],
    /// Pressure (from tablet, or 1.0 for mouse).
    pub pressure: f32,
    /// Timestamp within the stroke (seconds from start).
    pub time: f32,
}

/// A height sample stored for undo.
#[derive(Debug, Clone, Copy)]
pub struct HeightSample {
    /// Grid X coordinate.
    pub x: u32,
    /// Grid Z coordinate.
    pub z: u32,
    /// Height value before modification.
    pub old_height: f32,
    /// Height value after modification.
    pub new_height: f32,
}

/// A splatmap sample stored for undo.
#[derive(Debug, Clone, Copy)]
pub struct SplatmapSample {
    /// Grid X coordinate.
    pub x: u32,
    /// Grid Z coordinate.
    pub z: u32,
    /// Layer index.
    pub layer: u32,
    /// Weight before modification.
    pub old_weight: f32,
    /// Weight after modification.
    pub new_weight: f32,
}

impl TerrainBrushStroke {
    /// Create a new stroke.
    pub fn new(brush_type: BrushType, settings: BrushSettings) -> Self {
        Self {
            id: Uuid::new_v4(),
            brush_type,
            settings,
            points: Vec::new(),
            height_snapshot: Vec::new(),
            splatmap_snapshot: Vec::new(),
            completed: false,
        }
    }

    /// Add a point to the stroke.
    pub fn add_point(&mut self, position: [f32; 3], pressure: f32, time: f32) {
        self.points.push(StrokePoint {
            position,
            pressure,
            time,
        });
    }

    /// Mark the stroke as completed.
    pub fn complete(&mut self) {
        self.completed = true;
    }

    /// Record a height modification for undo.
    pub fn record_height(&mut self, x: u32, z: u32, old_height: f32, new_height: f32) {
        // Check if we already have a record for this cell (keep the original old_height).
        if let Some(existing) = self
            .height_snapshot
            .iter_mut()
            .find(|s| s.x == x && s.z == z)
        {
            existing.new_height = new_height;
        } else {
            self.height_snapshot.push(HeightSample {
                x,
                z,
                old_height,
                new_height,
            });
        }
    }

    /// Record a splatmap modification for undo.
    pub fn record_splatmap(
        &mut self,
        x: u32,
        z: u32,
        layer: u32,
        old_weight: f32,
        new_weight: f32,
    ) {
        if let Some(existing) = self
            .splatmap_snapshot
            .iter_mut()
            .find(|s| s.x == x && s.z == z && s.layer == layer)
        {
            existing.new_weight = new_weight;
        } else {
            self.splatmap_snapshot.push(SplatmapSample {
                x,
                z,
                layer,
                old_weight,
                new_weight,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Erosion Settings
// ---------------------------------------------------------------------------

/// Parameters for hydraulic erosion simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydraulicErosionSettings {
    /// Number of erosion iterations.
    pub iterations: u32,
    /// Amount of sediment a water droplet can carry.
    pub sediment_capacity: f32,
    /// Rate at which sediment is deposited.
    pub deposition_rate: f32,
    /// Rate at which terrain is eroded.
    pub erosion_rate: f32,
    /// Evaporation rate of water droplets.
    pub evaporation_rate: f32,
    /// Minimum slope before deposition occurs.
    pub min_slope: f32,
    /// Initial water volume per droplet.
    pub initial_water: f32,
    /// Inertia: how much the droplet resists changing direction.
    pub inertia: f32,
    /// Maximum steps per droplet lifetime.
    pub max_lifetime: u32,
    /// Gravity acceleration.
    pub gravity: f32,
}

impl Default for HydraulicErosionSettings {
    fn default() -> Self {
        Self {
            iterations: 50000,
            sediment_capacity: 4.0,
            deposition_rate: 0.3,
            erosion_rate: 0.3,
            evaporation_rate: 0.01,
            min_slope: 0.01,
            initial_water: 1.0,
            inertia: 0.05,
            max_lifetime: 64,
            gravity: 4.0,
        }
    }
}

/// Parameters for thermal erosion simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalErosionSettings {
    /// Number of erosion iterations.
    pub iterations: u32,
    /// Maximum slope angle (in degrees) before material slides.
    pub talus_angle: f32,
    /// Amount of material transferred per iteration.
    pub transfer_rate: f32,
}

impl Default for ThermalErosionSettings {
    fn default() -> Self {
        Self {
            iterations: 100,
            talus_angle: 45.0,
            transfer_rate: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Terrain Layer
// ---------------------------------------------------------------------------

/// A texture layer in the terrain's splatmap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainLayer {
    /// Layer display name.
    pub name: String,
    /// Diffuse/albedo texture asset UUID.
    pub albedo_texture: Option<Uuid>,
    /// Normal map texture asset UUID.
    pub normal_texture: Option<Uuid>,
    /// UV tiling scale.
    pub tiling: [f32; 2],
    /// Metallic value for this layer.
    pub metallic: f32,
    /// Roughness value for this layer.
    pub roughness: f32,
}

impl TerrainLayer {
    /// Create a new terrain layer.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            albedo_texture: None,
            normal_texture: None,
            tiling: [10.0, 10.0],
            metallic: 0.0,
            roughness: 0.8,
        }
    }
}

// ---------------------------------------------------------------------------
// Vegetation Type
// ---------------------------------------------------------------------------

/// A vegetation type that can be painted onto the terrain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegetationType {
    /// Display name.
    pub name: String,
    /// Mesh asset UUID.
    pub mesh: Option<Uuid>,
    /// Minimum scale.
    pub min_scale: f32,
    /// Maximum scale.
    pub max_scale: f32,
    /// Density (instances per square unit).
    pub density: f32,
    /// Minimum slope angle for placement.
    pub min_slope: f32,
    /// Maximum slope angle for placement.
    pub max_slope: f32,
    /// Random rotation enabled.
    pub random_rotation: bool,
    /// Align to terrain normal.
    pub align_to_normal: bool,
}

impl VegetationType {
    /// Create a new vegetation type.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            mesh: None,
            min_scale: 0.8,
            max_scale: 1.2,
            density: 1.0,
            min_slope: 0.0,
            max_slope: 45.0,
            random_rotation: true,
            align_to_normal: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Heightmap Stamp
// ---------------------------------------------------------------------------

/// A heightmap stamp that can be applied to the terrain.
#[derive(Debug, Clone)]
pub struct HeightmapStamp {
    /// Display name.
    pub name: String,
    /// Width of the stamp in pixels.
    pub width: u32,
    /// Height of the stamp in pixels.
    pub height: u32,
    /// Normalized height data [0.0, 1.0].
    pub data: Vec<f32>,
    /// Stamp scale (how much it affects height).
    pub scale: f32,
    /// Whether to blend additively or replace.
    pub blend_mode: StampBlendMode,
}

/// How a heightmap stamp blends with existing terrain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StampBlendMode {
    /// Add stamp values to existing height.
    Additive,
    /// Replace existing height with stamp values.
    Replace,
    /// Use maximum of existing and stamp.
    Max,
    /// Use minimum of existing and stamp.
    Min,
    /// Multiply existing height by stamp values.
    Multiply,
}

impl HeightmapStamp {
    /// Create a stamp from raw height data.
    pub fn new(name: impl Into<String>, width: u32, height: u32, data: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            width,
            height,
            data,
            scale: 1.0,
            blend_mode: StampBlendMode::Additive,
        }
    }

    /// Sample the stamp at normalized coordinates [0,1].
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let x = (u * (self.width - 1) as f32).round() as usize;
        let y = (v * (self.height - 1) as f32).round() as usize;
        let x = x.min(self.width as usize - 1);
        let y = y.min(self.height as usize - 1);
        let idx = y * self.width as usize + x;
        self.data.get(idx).copied().unwrap_or(0.0) * self.scale
    }

    /// Generate a simple cone stamp for testing.
    pub fn cone(size: u32) -> Self {
        let mut data = vec![0.0; (size * size) as usize];
        let center = size as f32 / 2.0;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dist = (dx * dx + dy * dy).sqrt() / center;
                data[(y * size + x) as usize] = (1.0 - dist).max(0.0);
            }
        }
        Self::new("Cone", size, size, data)
    }

    /// Generate a simple dome stamp.
    pub fn dome(size: u32) -> Self {
        let mut data = vec![0.0; (size * size) as usize];
        let center = size as f32 / 2.0;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dist = (dx * dx + dy * dy).sqrt() / center;
                let height = if dist < 1.0 {
                    (1.0 - dist * dist).sqrt()
                } else {
                    0.0
                };
                data[(y * size + x) as usize] = height;
            }
        }
        Self::new("Dome", size, size, data)
    }
}

// ---------------------------------------------------------------------------
// Terrain Editor
// ---------------------------------------------------------------------------

/// The terrain editor panel and tool state.
#[derive(Debug)]
pub struct TerrainEditor {
    /// Current brush type.
    pub brush_type: BrushType,
    /// Current brush settings.
    pub brush_settings: BrushSettings,
    /// Whether the editor is active/visible.
    pub active: bool,
    /// Current brush position in world space (follows cursor).
    pub cursor_position: [f32; 3],
    /// Whether the brush is currently being applied (mouse down).
    pub painting: bool,
    /// The active stroke being recorded.
    active_stroke: Option<TerrainBrushStroke>,
    /// Undo history of completed strokes.
    stroke_history: Vec<TerrainBrushStroke>,
    /// Redo stack.
    stroke_redo: Vec<TerrainBrushStroke>,
    /// Maximum undo history size.
    pub max_history: usize,
    /// Terrain texture layers.
    pub layers: Vec<TerrainLayer>,
    /// Vegetation types.
    pub vegetation_types: Vec<VegetationType>,
    /// Available heightmap stamps.
    pub stamps: Vec<HeightmapStamp>,
    /// Selected stamp index (for Stamp brush).
    pub selected_stamp: Option<usize>,
    /// Hydraulic erosion settings.
    pub hydraulic_erosion: HydraulicErosionSettings,
    /// Thermal erosion settings.
    pub thermal_erosion: ThermalErosionSettings,
    /// Elapsed time within the current stroke (for spacing calculation).
    stroke_elapsed: f32,
    /// Last applied position along the stroke (for spacing).
    last_apply_position: Option<[f32; 3]>,
}

impl Default for TerrainEditor {
    fn default() -> Self {
        let default_layers = vec![
            TerrainLayer::new("Grass"),
            TerrainLayer::new("Dirt"),
            TerrainLayer::new("Rock"),
            TerrainLayer::new("Sand"),
        ];

        let default_vegetation = vec![
            VegetationType::new("Grass Clump"),
            VegetationType::new("Tree"),
            VegetationType::new("Bush"),
        ];

        Self {
            brush_type: BrushType::Raise,
            brush_settings: BrushSettings::default(),
            active: false,
            cursor_position: [0.0, 0.0, 0.0],
            painting: false,
            active_stroke: None,
            stroke_history: Vec::new(),
            stroke_redo: Vec::new(),
            max_history: 50,
            layers: default_layers,
            vegetation_types: default_vegetation,
            stamps: vec![HeightmapStamp::cone(64), HeightmapStamp::dome(64)],
            selected_stamp: None,
            hydraulic_erosion: HydraulicErosionSettings::default(),
            thermal_erosion: ThermalErosionSettings::default(),
            stroke_elapsed: 0.0,
            last_apply_position: None,
        }
    }
}

impl TerrainEditor {
    /// Create a new terrain editor.
    pub fn new() -> Self {
        Self::default()
    }

    // --- Brush operations ---

    /// Begin a brush stroke at the current cursor position.
    pub fn begin_stroke(&mut self) {
        let mut stroke =
            TerrainBrushStroke::new(self.brush_type, self.brush_settings.clone());

        // For flatten brush, capture the starting height.
        if self.brush_type == BrushType::Flatten {
            self.brush_settings.flatten_height = Some(self.cursor_position[1]);
            stroke.settings.flatten_height = Some(self.cursor_position[1]);
        }

        stroke.add_point(self.cursor_position, 1.0, 0.0);
        self.active_stroke = Some(stroke);
        self.painting = true;
        self.stroke_elapsed = 0.0;
        self.last_apply_position = Some(self.cursor_position);
    }

    /// Update the brush stroke (called each frame while painting).
    pub fn update_stroke(&mut self, dt: f32) {
        if !self.painting {
            return;
        }

        self.stroke_elapsed += dt;

        // Check spacing requirement.
        let spacing_dist = self.brush_settings.size * self.brush_settings.spacing;
        let should_apply = if let Some(last_pos) = self.last_apply_position {
            let dx = self.cursor_position[0] - last_pos[0];
            let dz = self.cursor_position[2] - last_pos[2];
            let dist = (dx * dx + dz * dz).sqrt();
            dist >= spacing_dist
        } else {
            true
        };

        if should_apply {
            if let Some(ref mut stroke) = self.active_stroke {
                stroke.add_point(self.cursor_position, 1.0, self.stroke_elapsed);
            }
            self.last_apply_position = Some(self.cursor_position);
        }
    }

    /// End the current stroke.
    pub fn end_stroke(&mut self) {
        self.painting = false;
        if let Some(mut stroke) = self.active_stroke.take() {
            stroke.complete();
            self.stroke_history.push(stroke);
            self.stroke_redo.clear();

            // Trim history.
            while self.stroke_history.len() > self.max_history {
                self.stroke_history.remove(0);
            }
        }
        self.brush_settings.flatten_height = None;
        self.last_apply_position = None;
    }

    /// Cancel the current stroke (discard without saving).
    pub fn cancel_stroke(&mut self) {
        self.painting = false;
        self.active_stroke = None;
        self.last_apply_position = None;
    }

    // --- Undo/Redo ---

    /// Undo the last stroke.
    pub fn undo_stroke(&mut self) -> Option<&TerrainBrushStroke> {
        if let Some(stroke) = self.stroke_history.pop() {
            self.stroke_redo.push(stroke);
            self.stroke_redo.last()
        } else {
            None
        }
    }

    /// Redo the last undone stroke.
    pub fn redo_stroke(&mut self) -> Option<&TerrainBrushStroke> {
        if let Some(stroke) = self.stroke_redo.pop() {
            self.stroke_history.push(stroke);
            self.stroke_history.last()
        } else {
            None
        }
    }

    /// Whether undo is available.
    pub fn can_undo(&self) -> bool {
        !self.stroke_history.is_empty()
    }

    /// Whether redo is available.
    pub fn can_redo(&self) -> bool {
        !self.stroke_redo.is_empty()
    }

    // --- Layer management ---

    /// Add a terrain texture layer.
    pub fn add_layer(&mut self, layer: TerrainLayer) -> usize {
        self.layers.push(layer);
        self.layers.len() - 1
    }

    /// Remove a terrain layer by index.
    pub fn remove_layer(&mut self, index: usize) -> Option<TerrainLayer> {
        if index < self.layers.len() {
            Some(self.layers.remove(index))
        } else {
            None
        }
    }

    /// Add a vegetation type.
    pub fn add_vegetation_type(&mut self, veg: VegetationType) -> usize {
        self.vegetation_types.push(veg);
        self.vegetation_types.len() - 1
    }

    // --- Erosion ---

    /// Apply hydraulic erosion to a heightmap region.
    /// Returns the number of height samples modified.
    pub fn apply_hydraulic_erosion(&self, heightmap: &mut [f32], width: u32, height: u32) -> u32 {
        let settings = &self.hydraulic_erosion;
        let mut modified = 0_u32;
        let w = width as usize;
        let h = height as usize;

        for _ in 0..settings.iterations {
            // Random starting position.
            let mut pos_x = simple_hash_f32(modified) * (width - 2) as f32 + 1.0;
            let mut pos_y = simple_hash_f32(modified.wrapping_add(12345)) * (height - 2) as f32 + 1.0;
            let mut dir_x = 0.0_f32;
            let mut dir_y = 0.0_f32;
            let mut speed = 1.0_f32;
            let mut water = settings.initial_water;
            let mut sediment = 0.0_f32;

            for _ in 0..settings.max_lifetime {
                let ix = pos_x as usize;
                let iy = pos_y as usize;

                if ix < 1 || ix >= w - 1 || iy < 1 || iy >= h - 1 {
                    break;
                }

                // Compute gradient using bilinear interpolation of neighbors.
                let idx = iy * w + ix;
                let current_height = heightmap[idx];
                let grad_x = heightmap[idx + 1] - heightmap[idx.saturating_sub(1)];
                let grad_y = heightmap[idx + w] - heightmap[idx.saturating_sub(w)];

                // Update direction with inertia.
                dir_x = dir_x * settings.inertia - grad_x * (1.0 - settings.inertia);
                dir_y = dir_y * settings.inertia - grad_y * (1.0 - settings.inertia);

                let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt();
                if dir_len < 1e-8 {
                    break;
                }
                dir_x /= dir_len;
                dir_y /= dir_len;

                // Move droplet.
                pos_x += dir_x;
                pos_y += dir_y;

                let new_ix = pos_x as usize;
                let new_iy = pos_y as usize;
                if new_ix >= w || new_iy >= h {
                    break;
                }

                let new_height = heightmap[new_iy * w + new_ix];
                let height_diff = new_height - current_height;

                // Sediment capacity.
                let capacity = (-height_diff).max(settings.min_slope)
                    * speed
                    * water
                    * settings.sediment_capacity;

                if sediment > capacity || height_diff > 0.0 {
                    // Deposit sediment.
                    let deposit = if height_diff > 0.0 {
                        height_diff.min(sediment)
                    } else {
                        (sediment - capacity) * settings.deposition_rate
                    };
                    heightmap[idx] += deposit;
                    sediment -= deposit;
                    modified += 1;
                } else {
                    // Erode.
                    let erode = ((capacity - sediment) * settings.erosion_rate)
                        .min(-height_diff);
                    heightmap[idx] -= erode;
                    sediment += erode;
                    modified += 1;
                }

                speed = (speed * speed + height_diff.abs() * settings.gravity).sqrt();
                water *= 1.0 - settings.evaporation_rate;

                if water < 0.001 {
                    break;
                }
            }
        }

        modified
    }

    /// Apply thermal erosion to a heightmap region.
    pub fn apply_thermal_erosion(&self, heightmap: &mut [f32], width: u32, height: u32) -> u32 {
        let settings = &self.thermal_erosion;
        let w = width as usize;
        let h = height as usize;
        let talus = settings.talus_angle.to_radians().tan();
        let mut modified = 0_u32;

        let mut temp = heightmap.to_vec();

        for _ in 0..settings.iterations {
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let idx = y * w + x;
                    let center = heightmap[idx];

                    // Check all 4 neighbors.
                    let neighbors = [
                        (idx.wrapping_sub(1), 1.0_f32),
                        (idx + 1, 1.0),
                        (idx.wrapping_sub(w), 1.0),
                        (idx + w, 1.0),
                    ];

                    let mut max_diff = 0.0_f32;
                    let mut max_neighbor = idx;
                    let mut total_diff = 0.0_f32;

                    for &(ni, dist) in &neighbors {
                        if ni < heightmap.len() {
                            let diff = (center - heightmap[ni]) / dist;
                            if diff > talus {
                                total_diff += diff - talus;
                                if diff > max_diff {
                                    max_diff = diff;
                                    max_neighbor = ni;
                                }
                            }
                        }
                    }

                    if max_diff > talus && total_diff > 0.0 {
                        let transfer = (max_diff - talus) * settings.transfer_rate * 0.5;
                        temp[idx] -= transfer;
                        temp[max_neighbor] += transfer;
                        modified += 1;
                    }
                }
            }

            heightmap.copy_from_slice(&temp);
        }

        modified
    }

    // --- Rendering info ---

    /// Get brush visualization data for the renderer.
    pub fn brush_render_data(&self) -> BrushRenderData {
        BrushRenderData {
            position: self.cursor_position,
            radius: self.brush_settings.size,
            falloff: self.brush_settings.falloff,
            active: self.painting,
            brush_type: self.brush_type,
        }
    }

    /// Number of strokes in the undo history.
    pub fn history_count(&self) -> usize {
        self.stroke_history.len()
    }
}

/// Data for rendering the brush visualization.
#[derive(Debug, Clone)]
pub struct BrushRenderData {
    pub position: [f32; 3],
    pub radius: f32,
    pub falloff: BrushFalloff,
    pub active: bool,
    pub brush_type: BrushType,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple deterministic hash for pseudo-random number generation in erosion.
fn simple_hash_f32(seed: u32) -> f32 {
    let mut x = seed;
    x = x.wrapping_mul(0x45d9f3b);
    x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
    x = (x >> 16) ^ x;
    (x as f32) / (u32::MAX as f32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brush_falloff_values() {
        let settings = BrushSettings {
            size: 10.0,
            falloff: BrushFalloff::Linear,
            ..Default::default()
        };
        assert!((settings.compute_falloff(0.0) - 1.0).abs() < 1e-5);
        assert!((settings.compute_falloff(5.0) - 0.5).abs() < 1e-5);
        assert!((settings.compute_falloff(10.0) - 0.0).abs() < 1e-5);
        assert!((settings.compute_falloff(15.0) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn brush_falloff_constant() {
        let settings = BrushSettings {
            size: 10.0,
            falloff: BrushFalloff::Constant,
            ..Default::default()
        };
        assert!((settings.compute_falloff(0.0) - 1.0).abs() < 1e-5);
        assert!((settings.compute_falloff(9.9) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn stroke_recording() {
        let mut stroke = TerrainBrushStroke::new(BrushType::Raise, BrushSettings::default());
        stroke.add_point([0.0, 0.0, 0.0], 1.0, 0.0);
        stroke.add_point([1.0, 0.0, 1.0], 1.0, 0.016);
        assert_eq!(stroke.points.len(), 2);
        assert!(!stroke.completed);
        stroke.complete();
        assert!(stroke.completed);
    }

    #[test]
    fn stroke_height_snapshot() {
        let mut stroke = TerrainBrushStroke::new(BrushType::Raise, BrushSettings::default());
        stroke.record_height(5, 5, 10.0, 11.0);
        stroke.record_height(5, 5, 10.0, 12.0); // Should update, keeping old_height=10.
        assert_eq!(stroke.height_snapshot.len(), 1);
        assert!((stroke.height_snapshot[0].old_height - 10.0).abs() < 1e-5);
        assert!((stroke.height_snapshot[0].new_height - 12.0).abs() < 1e-5);
    }

    #[test]
    fn terrain_editor_stroke_workflow() {
        let mut editor = TerrainEditor::new();
        editor.cursor_position = [5.0, 0.0, 5.0];

        editor.begin_stroke();
        assert!(editor.painting);

        editor.cursor_position = [6.0, 0.0, 6.0];
        editor.update_stroke(0.016);

        editor.end_stroke();
        assert!(!editor.painting);
        assert_eq!(editor.history_count(), 1);
    }

    #[test]
    fn terrain_editor_undo_redo() {
        let mut editor = TerrainEditor::new();

        // Create two strokes.
        editor.begin_stroke();
        editor.end_stroke();
        editor.begin_stroke();
        editor.end_stroke();
        assert_eq!(editor.history_count(), 2);

        assert!(editor.can_undo());
        editor.undo_stroke();
        assert_eq!(editor.history_count(), 1);
        assert!(editor.can_redo());

        editor.redo_stroke();
        assert_eq!(editor.history_count(), 2);
        assert!(!editor.can_redo());
    }

    #[test]
    fn terrain_layer_management() {
        let mut editor = TerrainEditor::new();
        let initial_layers = editor.layers.len();
        editor.add_layer(TerrainLayer::new("Snow"));
        assert_eq!(editor.layers.len(), initial_layers + 1);

        let removed = editor.remove_layer(0);
        assert!(removed.is_some());
        assert_eq!(editor.layers.len(), initial_layers);
    }

    #[test]
    fn heightmap_stamp_sample() {
        let stamp = HeightmapStamp::cone(16);
        let center = stamp.sample(0.5, 0.5);
        let edge = stamp.sample(0.0, 0.5);
        assert!(center > edge);
        assert!(center > 0.0);
    }

    #[test]
    fn thermal_erosion_basic() {
        let editor = TerrainEditor::new();
        let mut heightmap = vec![0.0; 16 * 16];
        // Create a spike.
        heightmap[8 * 16 + 8] = 100.0;

        let modified = editor.apply_thermal_erosion(&mut heightmap, 16, 16);
        assert!(modified > 0);
        // The spike should be reduced.
        assert!(heightmap[8 * 16 + 8] < 100.0);
    }

    #[test]
    fn brush_render_data() {
        let editor = TerrainEditor::new();
        let data = editor.brush_render_data();
        assert!(!data.active);
        assert!(data.radius > 0.0);
    }

    #[test]
    fn dome_stamp() {
        let stamp = HeightmapStamp::dome(32);
        let center = stamp.sample(0.5, 0.5);
        assert!(center > 0.9); // Center of dome should be close to 1.0
    }
}
