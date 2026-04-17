// engine/render/src/terrain_rendering_v2.rs
//
// Enhanced terrain rendering: virtual texturing with clipmap LOD,
// tessellation-based detail, procedural snow accumulation, puddle
// simulation from rain, and material blending with height-based layering.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Core math types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 { pub x: f32, pub y: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
    pub fn length(self) -> f32 { (self.x*self.x + self.y*self.y).sqrt() }
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self { x: self.y*r.z - self.z*r.y, y: self.z*r.x - self.x*r.z, z: self.x*r.y - self.y*r.x } }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn normalize(self) -> Self { let l = self.length(); if l < 1e-12 { Self::ZERO } else { Self { x:self.x/l, y:self.y/l, z:self.z/l } } }
    pub fn scale(self, s: f32) -> Self { Self { x:self.x*s, y:self.y*s, z:self.z*s } }
    pub fn add(self, r: Self) -> Self { Self { x:self.x+r.x, y:self.y+r.y, z:self.z+r.z } }
    pub fn sub(self, r: Self) -> Self { Self { x:self.x-r.x, y:self.y-r.y, z:self.z-r.z } }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
}

// ---------------------------------------------------------------------------
// Heightmap data
// ---------------------------------------------------------------------------

/// Raw heightmap data for terrain.
#[derive(Debug, Clone)]
pub struct Heightmap {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
    pub min_height: f32,
    pub max_height: f32,
    pub world_scale: Vec3, // scale factor for world units per texel
}

impl Heightmap {
    pub fn new(width: u32, height: u32, scale: Vec3) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; (width * height) as usize],
            min_height: 0.0,
            max_height: 1.0,
            world_scale: scale,
        }
    }

    /// Sample height at a normalized UV coordinate with bilinear interpolation.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> f32 {
        let fx = u * (self.width - 1) as f32;
        let fy = v * (self.height - 1) as f32;
        let x0 = (fx.floor() as u32).min(self.width - 2);
        let y0 = (fy.floor() as u32).min(self.height - 2);
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let h00 = self.data[(y0 * self.width + x0) as usize];
        let h10 = self.data[(y0 * self.width + x1) as usize];
        let h01 = self.data[(y1 * self.width + x0) as usize];
        let h11 = self.data[(y1 * self.width + x1) as usize];

        let h0 = h00 + tx * (h10 - h00);
        let h1 = h01 + tx * (h11 - h01);
        h0 + ty * (h1 - h0)
    }

    /// Compute surface normal at a given texel position.
    pub fn compute_normal(&self, x: u32, y: u32) -> Vec3 {
        let x0 = if x > 0 { x - 1 } else { 0 };
        let x1 = (x + 1).min(self.width - 1);
        let y0 = if y > 0 { y - 1 } else { 0 };
        let y1 = (y + 1).min(self.height - 1);

        let hl = self.data[(y * self.width + x0) as usize];
        let hr = self.data[(y * self.width + x1) as usize];
        let hd = self.data[(y0 * self.width + x) as usize];
        let hu = self.data[(y1 * self.width + x) as usize];

        let dx = (hr - hl) * self.world_scale.y / (2.0 * self.world_scale.x);
        let dz = (hu - hd) * self.world_scale.y / (2.0 * self.world_scale.z);

        Vec3::new(-dx, 1.0, -dz).normalize()
    }

    /// Compute slope angle in radians at a texel.
    pub fn slope_at(&self, x: u32, y: u32) -> f32 {
        let normal = self.compute_normal(x, y);
        normal.dot(Vec3::UP).acos()
    }

    /// World position of a texel.
    pub fn world_position(&self, x: u32, y: u32) -> Vec3 {
        let h = self.data[(y * self.width + x) as usize];
        Vec3::new(
            x as f32 * self.world_scale.x,
            h * self.world_scale.y,
            y as f32 * self.world_scale.z,
        )
    }

    /// Generate the full normal map.
    pub fn compute_normal_map(&self) -> Vec<Vec3> {
        let mut normals = Vec::with_capacity((self.width * self.height) as usize);
        for y in 0..self.height {
            for x in 0..self.width {
                normals.push(self.compute_normal(x, y));
            }
        }
        normals
    }
}

// ---------------------------------------------------------------------------
// Clipmap system
// ---------------------------------------------------------------------------

/// Clipmap level for terrain LOD.
#[derive(Debug, Clone)]
pub struct ClipmapLevel {
    /// Level index (0 = finest).
    pub level: u32,
    /// Size of this clipmap in texels (power of 2).
    pub size: u32,
    /// World-space extent covered by one texel at this level.
    pub texel_size: f32,
    /// Center of the clipmap in world space (snapped to texel grid).
    pub center: Vec2,
    /// Height data for this level.
    pub data: Vec<f32>,
    /// Normal data for this level.
    pub normals: Vec<Vec3>,
    /// Whether this level needs an update.
    pub dirty: bool,
}

/// Clipmap terrain rendering system.
#[derive(Debug)]
pub struct ClipmapTerrain {
    pub levels: Vec<ClipmapLevel>,
    pub heightmap: Heightmap,
    pub num_levels: u32,
    pub clipmap_size: u32, // texels per side at each level
    pub base_texel_size: f32,
}

impl ClipmapTerrain {
    /// Create a new clipmap terrain.
    pub fn new(heightmap: Heightmap, num_levels: u32, clipmap_size: u32) -> Self {
        let base_texel_size = heightmap.world_scale.x;
        let mut levels = Vec::with_capacity(num_levels as usize);

        for level in 0..num_levels {
            let texel_size = base_texel_size * (1 << level) as f32;
            let total_texels = (clipmap_size * clipmap_size) as usize;
            levels.push(ClipmapLevel {
                level,
                size: clipmap_size,
                texel_size,
                center: Vec2::ZERO,
                data: vec![0.0; total_texels],
                normals: vec![Vec3::UP; total_texels],
                dirty: true,
            });
        }

        Self {
            levels,
            heightmap,
            num_levels,
            clipmap_size,
            base_texel_size,
        }
    }

    /// Update clipmap centers based on camera position.
    pub fn update(&mut self, camera_pos: Vec3) {
        for level in &mut self.levels {
            let snap = level.texel_size;
            let new_center = Vec2::new(
                (camera_pos.x / snap).round() * snap,
                (camera_pos.z / snap).round() * snap,
            );

            if (new_center.x - level.center.x).abs() >= snap
                || (new_center.y - level.center.y).abs() >= snap
            {
                level.center = new_center;
                level.dirty = true;
            }
        }

        // Refill dirty levels
        for i in 0..self.levels.len() {
            if self.levels[i].dirty {
                self.fill_level(i);
                self.levels[i].dirty = false;
            }
        }
    }

    /// Fill a clipmap level by sampling from the heightmap.
    fn fill_level(&mut self, level_idx: usize) {
        let level = &self.levels[level_idx];
        let half_size = level.size as f32 * level.texel_size * 0.5;
        let origin_x = level.center.x - half_size;
        let origin_z = level.center.y - half_size;
        let hm_width = self.heightmap.width as f32 * self.heightmap.world_scale.x;
        let hm_depth = self.heightmap.height as f32 * self.heightmap.world_scale.z;

        let size = level.size;
        let texel_size = level.texel_size;

        // We need to write to levels[level_idx], but we also read from heightmap
        for y in 0..size {
            for x in 0..size {
                let wx = origin_x + x as f32 * texel_size;
                let wz = origin_z + y as f32 * texel_size;

                let u = (wx / hm_width).clamp(0.0, 1.0);
                let v = (wz / hm_depth).clamp(0.0, 1.0);

                let idx = (y * size + x) as usize;
                self.levels[level_idx].data[idx] = self.heightmap.sample_bilinear(u, v);

                // Compute normal from neighbors
                let u_left = ((wx - texel_size) / hm_width).clamp(0.0, 1.0);
                let u_right = ((wx + texel_size) / hm_width).clamp(0.0, 1.0);
                let v_down = ((wz - texel_size) / hm_depth).clamp(0.0, 1.0);
                let v_up = ((wz + texel_size) / hm_depth).clamp(0.0, 1.0);

                let hl = self.heightmap.sample_bilinear(u_left, v);
                let hr = self.heightmap.sample_bilinear(u_right, v);
                let hd = self.heightmap.sample_bilinear(u, v_down);
                let hu = self.heightmap.sample_bilinear(u, v_up);

                let dx = (hr - hl) * self.heightmap.world_scale.y / (2.0 * texel_size);
                let dz = (hu - hd) * self.heightmap.world_scale.y / (2.0 * texel_size);

                self.levels[level_idx].normals[idx] = Vec3::new(-dx, 1.0, -dz).normalize();
            }
        }
    }

    /// Select which LOD level to render for a given distance from camera.
    pub fn select_lod(&self, distance: f32) -> u32 {
        for i in 0..self.num_levels {
            let threshold = self.base_texel_size * (1 << i) as f32 * self.clipmap_size as f32 * 0.3;
            if distance < threshold {
                return i;
            }
        }
        self.num_levels - 1
    }

    /// Generate mesh patches for rendering the terrain.
    pub fn generate_patches(&self, camera_pos: Vec3) -> Vec<TerrainPatch> {
        let mut patches = Vec::new();
        let patch_size = 16_u32; // vertices per patch side

        for level in &self.levels {
            let patches_per_side = level.size / patch_size;
            let half_size = level.size as f32 * level.texel_size * 0.5;

            for py in 0..patches_per_side {
                for px in 0..patches_per_side {
                    let patch_x = level.center.x - half_size + px as f32 * patch_size as f32 * level.texel_size;
                    let patch_z = level.center.y - half_size + py as f32 * patch_size as f32 * level.texel_size;
                    let patch_center = Vec3::new(
                        patch_x + patch_size as f32 * level.texel_size * 0.5,
                        0.0,
                        patch_z + patch_size as f32 * level.texel_size * 0.5,
                    );

                    let dist = Vec2::new(
                        camera_pos.x - patch_center.x,
                        camera_pos.z - patch_center.z,
                    ).length();

                    // Skip patches covered by a finer level
                    if level.level > 0 {
                        let finer = &self.levels[(level.level - 1) as usize];
                        let finer_half = finer.size as f32 * finer.texel_size * 0.5;
                        if (patch_center.x - finer.center.x).abs() < finer_half
                            && (patch_center.z - finer.center.y).abs() < finer_half
                        {
                            continue;
                        }
                    }

                    patches.push(TerrainPatch {
                        world_x: patch_x,
                        world_z: patch_z,
                        size: patch_size as f32 * level.texel_size,
                        lod_level: level.level,
                        distance: dist,
                        tessellation_factor: compute_tessellation(dist, level.texel_size),
                    });
                }
            }
        }

        patches
    }
}

fn compute_tessellation(distance: f32, texel_size: f32) -> f32 {
    let target_screen_size = 8.0; // pixels per triangle edge
    let factor = (texel_size * 100.0 / distance.max(1.0)).clamp(1.0, 64.0);
    factor
}

/// A patch of terrain ready for rendering.
#[derive(Debug, Clone)]
pub struct TerrainPatch {
    pub world_x: f32,
    pub world_z: f32,
    pub size: f32,
    pub lod_level: u32,
    pub distance: f32,
    pub tessellation_factor: f32,
}

// ---------------------------------------------------------------------------
// Virtual texturing
// ---------------------------------------------------------------------------

/// Virtual texture page table for terrain materials.
#[derive(Debug)]
pub struct VirtualTextureSystem {
    pub page_size: u32,           // e.g., 128x128
    pub physical_pages_x: u32,   // physical texture atlas width in pages
    pub physical_pages_y: u32,   // physical texture atlas height in pages
    pub page_table: Vec<PageTableEntry>,
    pub page_table_width: u32,
    pub page_table_height: u32,
    pub lru_cache: Vec<CachedPage>,
    pub pending_requests: Vec<PageRequest>,
    pub max_pending: usize,
    pub frame_budget: u32,       // max pages to load per frame
}

#[derive(Debug, Clone, Copy)]
pub struct PageTableEntry {
    pub physical_x: u16,
    pub physical_y: u16,
    pub mip_level: u8,
    pub valid: bool,
}

impl Default for PageTableEntry {
    fn default() -> Self {
        Self { physical_x: 0, physical_y: 0, mip_level: 0, valid: false }
    }
}

#[derive(Debug, Clone)]
pub struct CachedPage {
    pub virtual_x: u32,
    pub virtual_y: u32,
    pub mip_level: u32,
    pub physical_x: u32,
    pub physical_y: u32,
    pub last_used_frame: u64,
    pub priority: f32,
}

#[derive(Debug, Clone)]
pub struct PageRequest {
    pub virtual_x: u32,
    pub virtual_y: u32,
    pub mip_level: u32,
    pub priority: f32,
}

impl VirtualTextureSystem {
    pub fn new(page_size: u32, physical_x: u32, physical_y: u32, virtual_width: u32, virtual_height: u32) -> Self {
        let pt_w = virtual_width / page_size;
        let pt_h = virtual_height / page_size;
        Self {
            page_size,
            physical_pages_x: physical_x,
            physical_pages_y: physical_y,
            page_table: vec![PageTableEntry::default(); (pt_w * pt_h) as usize],
            page_table_width: pt_w,
            page_table_height: pt_h,
            lru_cache: Vec::new(),
            pending_requests: Vec::new(),
            max_pending: 256,
            frame_budget: 8,
        }
    }

    /// Request a virtual texture page.
    pub fn request_page(&mut self, vx: u32, vy: u32, mip: u32, priority: f32) {
        // Check if already loaded
        let idx = (vy * self.page_table_width + vx) as usize;
        if idx < self.page_table.len() && self.page_table[idx].valid {
            return;
        }

        // Check if already pending
        if self.pending_requests.iter().any(|r| r.virtual_x == vx && r.virtual_y == vy && r.mip_level == mip) {
            return;
        }

        if self.pending_requests.len() < self.max_pending {
            self.pending_requests.push(PageRequest {
                virtual_x: vx,
                virtual_y: vy,
                mip_level: mip,
                priority,
            });
        }
    }

    /// Process pending page requests (called once per frame).
    pub fn update(&mut self, frame: u64) {
        // Sort by priority (highest first)
        self.pending_requests.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        let budget = self.frame_budget as usize;
        let to_process: Vec<_> = self.pending_requests.drain(..budget.min(self.pending_requests.len())).collect();

        for req in to_process {
            if let Some((px, py)) = self.allocate_physical_page(frame) {
                let idx = (req.virtual_y * self.page_table_width + req.virtual_x) as usize;
                if idx < self.page_table.len() {
                    self.page_table[idx] = PageTableEntry {
                        physical_x: px as u16,
                        physical_y: py as u16,
                        mip_level: req.mip_level as u8,
                        valid: true,
                    };
                }

                self.lru_cache.push(CachedPage {
                    virtual_x: req.virtual_x,
                    virtual_y: req.virtual_y,
                    mip_level: req.mip_level,
                    physical_x: px,
                    physical_y: py,
                    last_used_frame: frame,
                    priority: req.priority,
                });
            }
        }
    }

    /// Allocate a physical page, evicting the least-recently-used if needed.
    fn allocate_physical_page(&mut self, frame: u64) -> Option<(u32, u32)> {
        let max_pages = self.physical_pages_x * self.physical_pages_y;

        if self.lru_cache.len() < max_pages as usize {
            // Still have free pages
            let idx = self.lru_cache.len() as u32;
            let x = idx % self.physical_pages_x;
            let y = idx / self.physical_pages_x;
            return Some((x, y));
        }

        // Evict LRU page
        if let Some(lru_idx) = self.lru_cache.iter()
            .enumerate()
            .min_by_key(|(_, p)| p.last_used_frame)
            .map(|(i, _)| i)
        {
            let evicted = self.lru_cache.remove(lru_idx);

            // Invalidate the page table entry
            let pt_idx = (evicted.virtual_y * self.page_table_width + evicted.virtual_x) as usize;
            if pt_idx < self.page_table.len() {
                self.page_table[pt_idx].valid = false;
            }

            return Some((evicted.physical_x, evicted.physical_y));
        }

        None
    }

    /// Touch a page to update its LRU timestamp.
    pub fn touch_page(&mut self, vx: u32, vy: u32, frame: u64) {
        for page in &mut self.lru_cache {
            if page.virtual_x == vx && page.virtual_y == vy {
                page.last_used_frame = frame;
                return;
            }
        }
    }

    /// Compute the UV for the physical texture atlas.
    pub fn virtual_to_physical_uv(&self, virtual_u: f32, virtual_v: f32) -> Option<Vec2> {
        let vx = (virtual_u * self.page_table_width as f32) as u32;
        let vy = (virtual_v * self.page_table_height as f32) as u32;
        let idx = (vy * self.page_table_width + vx) as usize;

        if idx < self.page_table.len() && self.page_table[idx].valid {
            let entry = &self.page_table[idx];
            let frac_u = virtual_u * self.page_table_width as f32 - vx as f32;
            let frac_v = virtual_v * self.page_table_height as f32 - vy as f32;

            let pu = (entry.physical_x as f32 + frac_u) / self.physical_pages_x as f32;
            let pv = (entry.physical_y as f32 + frac_v) / self.physical_pages_y as f32;

            Some(Vec2::new(pu, pv))
        } else {
            None
        }
    }

    pub fn stats(&self) -> VirtualTextureStats {
        let total = self.physical_pages_x * self.physical_pages_y;
        let used = self.lru_cache.len() as u32;
        VirtualTextureStats {
            total_physical_pages: total,
            used_physical_pages: used,
            pending_requests: self.pending_requests.len() as u32,
            page_table_entries: self.page_table.len() as u32,
            valid_entries: self.page_table.iter().filter(|e| e.valid).count() as u32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VirtualTextureStats {
    pub total_physical_pages: u32,
    pub used_physical_pages: u32,
    pub pending_requests: u32,
    pub page_table_entries: u32,
    pub valid_entries: u32,
}

// ---------------------------------------------------------------------------
// Terrain material layers
// ---------------------------------------------------------------------------

/// A single terrain material layer.
#[derive(Debug, Clone)]
pub struct TerrainLayer {
    pub name: String,
    pub albedo_color: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    pub normal_strength: f32,
    pub height_blend_sharpness: f32,
    pub tiling: Vec2,
    pub min_height: f32,
    pub max_height: f32,
    pub min_slope: f32,  // radians
    pub max_slope: f32,  // radians
}

impl Default for TerrainLayer {
    fn default() -> Self {
        Self {
            name: String::new(),
            albedo_color: Vec3::new(0.5, 0.5, 0.5),
            roughness: 0.8,
            metallic: 0.0,
            normal_strength: 1.0,
            height_blend_sharpness: 4.0,
            tiling: Vec2::new(1.0, 1.0),
            min_height: f32::NEG_INFINITY,
            max_height: f32::INFINITY,
            min_slope: 0.0,
            max_slope: PI * 0.5,
        }
    }
}

/// Terrain material blending system.
pub struct TerrainMaterialBlender {
    pub layers: Vec<TerrainLayer>,
}

impl TerrainMaterialBlender {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: TerrainLayer) -> usize {
        let idx = self.layers.len();
        self.layers.push(layer);
        idx
    }

    /// Compute blend weights for all layers at a given position.
    pub fn compute_weights(&self, height: f32, slope: f32, layer_heights: &[f32]) -> Vec<f32> {
        let mut weights = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let mut w = 1.0_f32;

            // Height-based weight
            if height < layer.min_height || height > layer.max_height {
                w = 0.0;
            } else {
                let height_range = layer.max_height - layer.min_height;
                if height_range > 0.0 {
                    let center = (layer.min_height + layer.max_height) * 0.5;
                    let dist = (height - center).abs() / (height_range * 0.5);
                    w *= (1.0 - dist).max(0.0);
                }
            }

            // Slope-based weight
            if slope < layer.min_slope || slope > layer.max_slope {
                w = 0.0;
            } else {
                let slope_range = layer.max_slope - layer.min_slope;
                if slope_range > 0.0 {
                    let center = (layer.min_slope + layer.max_slope) * 0.5;
                    let dist = (slope - center).abs() / (slope_range * 0.5);
                    w *= (1.0 - dist).max(0.0);
                }
            }

            // Height-based sharpening
            if i < layer_heights.len() {
                let height_offset = layer_heights[i];
                w *= (1.0 + height_offset * layer.height_blend_sharpness).max(0.0);
            }

            weights.push(w);
        }

        // Normalize weights
        let total: f32 = weights.iter().sum();
        if total > 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }

        weights
    }

    /// Blend material properties based on weights.
    pub fn blend_material(&self, weights: &[f32]) -> BlendedMaterial {
        let mut albedo = Vec3::ZERO;
        let mut roughness = 0.0_f32;
        let mut metallic = 0.0_f32;

        for (i, layer) in self.layers.iter().enumerate() {
            if i >= weights.len() { break; }
            let w = weights[i];
            albedo = albedo.add(layer.albedo_color.scale(w));
            roughness += layer.roughness * w;
            metallic += layer.metallic * w;
        }

        BlendedMaterial { albedo, roughness, metallic }
    }
}

#[derive(Debug, Clone)]
pub struct BlendedMaterial {
    pub albedo: Vec3,
    pub roughness: f32,
    pub metallic: f32,
}

// ---------------------------------------------------------------------------
// Snow accumulation
// ---------------------------------------------------------------------------

/// Snow accumulation simulation on terrain.
pub struct SnowAccumulation {
    pub width: u32,
    pub height: u32,
    pub accumulation: Vec<f32>,  // snow depth per texel
    pub max_depth: f32,
    pub accumulation_rate: f32,  // units per second during snowfall
    pub melt_rate: f32,          // units per second when melting
    pub slope_threshold: f32,    // max slope for snow (radians)
    pub temperature: f32,        // current temperature
    pub snowfall_intensity: f32, // 0..1
    pub wind_direction: Vec2,
    pub wind_drift_strength: f32,
}

impl SnowAccumulation {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            accumulation: vec![0.0; (width * height) as usize],
            max_depth: 1.0,
            accumulation_rate: 0.01,
            melt_rate: 0.005,
            slope_threshold: 0.8, // ~45 degrees
            temperature: -5.0,
            snowfall_intensity: 0.5,
            wind_direction: Vec2::new(1.0, 0.0),
            wind_drift_strength: 0.3,
        }
    }

    /// Update snow accumulation for one frame.
    pub fn update(&mut self, dt: f32, heightmap: &Heightmap) {
        let width = self.width;
        let height = self.height;

        // Temporary buffer for wind drift
        let mut drift = vec![0.0_f32; (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let hm_x = (x as f32 / width as f32 * heightmap.width as f32) as u32;
                let hm_y = (y as f32 / height as f32 * heightmap.height as f32) as u32;
                let hm_x = hm_x.min(heightmap.width - 1);
                let hm_y = hm_y.min(heightmap.height - 1);

                let slope = heightmap.slope_at(hm_x, hm_y);
                let normal = heightmap.compute_normal(hm_x, hm_y);

                // Snowfall (only on slopes below threshold)
                if self.snowfall_intensity > 0.0 && slope < self.slope_threshold && self.temperature < 0.0 {
                    let slope_factor = 1.0 - (slope / self.slope_threshold);
                    let upward_factor = normal.dot(Vec3::UP).max(0.0);
                    self.accumulation[idx] += self.accumulation_rate * self.snowfall_intensity
                        * slope_factor * upward_factor * dt;
                }

                // Melting
                if self.temperature > 0.0 && self.accumulation[idx] > 0.0 {
                    let melt = self.melt_rate * self.temperature * dt;
                    self.accumulation[idx] = (self.accumulation[idx] - melt).max(0.0);
                }

                // Wind drift: move snow in wind direction
                if self.accumulation[idx] > 0.0 && self.wind_drift_strength > 0.0 {
                    let drift_amount = self.accumulation[idx] * self.wind_drift_strength * dt;
                    let dx = (self.wind_direction.x * 1.5).round() as i32;
                    let dy = (self.wind_direction.y * 1.5).round() as i32;
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                    let nidx = (ny * width + nx) as usize;

                    drift[idx] -= drift_amount;
                    drift[nidx] += drift_amount;
                }

                self.accumulation[idx] = self.accumulation[idx].clamp(0.0, self.max_depth);
            }
        }

        // Apply drift
        for i in 0..self.accumulation.len() {
            self.accumulation[i] = (self.accumulation[i] + drift[i]).clamp(0.0, self.max_depth);
        }
    }

    /// Get snow amount at a UV coordinate.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let x = (u * (self.width - 1) as f32).round() as u32;
        let y = (v * (self.height - 1) as f32).round() as u32;
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        self.accumulation[(y * self.width + x) as usize]
    }

    /// Get snow visual parameters for rendering.
    pub fn snow_material_params(&self, snow_depth: f32) -> SnowVisual {
        let coverage = (snow_depth / 0.1).clamp(0.0, 1.0);
        SnowVisual {
            coverage,
            albedo: Vec3::new(0.95, 0.95, 0.98),
            roughness: 0.4 + 0.4 * (1.0 - coverage), // fresh snow is smoother
            sparkle_intensity: coverage * 0.3,
            normal_flatten: coverage * 0.8, // snow flattens surface normals
        }
    }
}

#[derive(Debug, Clone)]
pub struct SnowVisual {
    pub coverage: f32,
    pub albedo: Vec3,
    pub roughness: f32,
    pub sparkle_intensity: f32,
    pub normal_flatten: f32,
}

// ---------------------------------------------------------------------------
// Puddle system
// ---------------------------------------------------------------------------

/// Rain puddle simulation on terrain.
pub struct PuddleSystem {
    pub width: u32,
    pub height: u32,
    pub water_depth: Vec<f32>,
    pub rain_intensity: f32,      // 0..1
    pub drain_rate: f32,          // depth per second
    pub evaporation_rate: f32,    // depth per second
    pub max_puddle_depth: f32,
    pub ripple_time: f32,         // for animated ripples
    pub surface_wetness: Vec<f32>,
}

impl PuddleSystem {
    pub fn new(width: u32, height: u32) -> Self {
        let total = (width * height) as usize;
        Self {
            width,
            height,
            water_depth: vec![0.0; total],
            rain_intensity: 0.0,
            drain_rate: 0.001,
            evaporation_rate: 0.0005,
            max_puddle_depth: 0.05,
            ripple_time: 0.0,
            surface_wetness: vec![0.0; total],
        }
    }

    /// Update puddle simulation.
    pub fn update(&mut self, dt: f32, heightmap: &Heightmap) {
        self.ripple_time += dt;

        let width = self.width;
        let height = self.height;

        // Flow simulation (simplified)
        let mut flow = vec![0.0_f32; (width * height) as usize];

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = (y * width + x) as usize;

                // Add rain
                if self.rain_intensity > 0.0 {
                    self.water_depth[idx] += self.rain_intensity * 0.001 * dt;
                }

                // Drain and evaporate
                self.water_depth[idx] -= (self.drain_rate + self.evaporation_rate) * dt;
                self.water_depth[idx] = self.water_depth[idx].max(0.0);

                // Flow to lower neighbors
                let hm_x = (x as f32 / width as f32 * heightmap.width as f32) as u32;
                let hm_y = (y as f32 / height as f32 * heightmap.height as f32) as u32;
                let hm_x = hm_x.min(heightmap.width - 1);
                let hm_y = hm_y.min(heightmap.height - 1);

                let h_center = heightmap.data[(hm_y * heightmap.width + hm_x) as usize]
                    + self.water_depth[idx];

                // Check 4 neighbors
                let neighbors = [
                    ((x + 1) as usize + (y as usize) * width as usize, 1i32, 0i32),
                    ((x - 1) as usize + (y as usize) * width as usize, -1, 0),
                    ((x as usize) + ((y + 1) as usize) * width as usize, 0, 1),
                    ((x as usize) + ((y - 1) as usize) * width as usize, 0, -1),
                ];

                for &(nidx, dx, dy) in &neighbors {
                    let nx = (hm_x as i32 + dx).clamp(0, heightmap.width as i32 - 1) as u32;
                    let ny = (hm_y as i32 + dy).clamp(0, heightmap.height as i32 - 1) as u32;
                    let h_neighbor = heightmap.data[(ny * heightmap.width + nx) as usize]
                        + self.water_depth.get(nidx).copied().unwrap_or(0.0);

                    if h_center > h_neighbor {
                        let diff = (h_center - h_neighbor) * 0.25 * dt;
                        let transfer = diff.min(self.water_depth[idx] * 0.25);
                        flow[idx] -= transfer;
                        if nidx < flow.len() {
                            flow[nidx] += transfer;
                        }
                    }
                }

                // Update surface wetness
                if self.water_depth[idx] > 0.001 || self.rain_intensity > 0.0 {
                    self.surface_wetness[idx] = (self.surface_wetness[idx] + dt * 2.0).min(1.0);
                } else {
                    self.surface_wetness[idx] = (self.surface_wetness[idx] - dt * 0.2).max(0.0);
                }
            }
        }

        // Apply flow
        for i in 0..self.water_depth.len() {
            self.water_depth[i] = (self.water_depth[i] + flow[i]).clamp(0.0, self.max_puddle_depth);
        }
    }

    /// Get puddle visual parameters.
    pub fn puddle_visual(&self, u: f32, v: f32) -> PuddleVisual {
        let x = (u * (self.width - 1) as f32).round() as u32;
        let y = (v * (self.height - 1) as f32).round() as u32;
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        let idx = (y * self.width + x) as usize;

        let depth = self.water_depth[idx];
        let wetness = self.surface_wetness[idx];
        let has_puddle = depth > 0.002;

        // Ripple animation
        let ripple_intensity = if has_puddle && self.rain_intensity > 0.0 {
            self.rain_intensity * 0.5
        } else {
            0.0
        };

        PuddleVisual {
            water_depth: depth,
            surface_wetness: wetness,
            roughness_mod: -wetness * 0.6, // wet surfaces are smoother
            darkening: wetness * 0.3,       // wet surfaces are darker
            ripple_intensity,
            ripple_time: self.ripple_time,
            reflectivity: if has_puddle { 0.8 } else { wetness * 0.3 },
        }
    }
}

#[derive(Debug, Clone)]
pub struct PuddleVisual {
    pub water_depth: f32,
    pub surface_wetness: f32,
    pub roughness_mod: f32,
    pub darkening: f32,
    pub ripple_intensity: f32,
    pub ripple_time: f32,
    pub reflectivity: f32,
}

// ---------------------------------------------------------------------------
// Procedural detail
// ---------------------------------------------------------------------------

/// Procedural detail for close-up terrain rendering.
pub struct ProceduralTerrainDetail {
    pub detail_layers: Vec<DetailLayer>,
    pub view_distance: f32,
    pub density_map_size: u32,
}

#[derive(Debug, Clone)]
pub struct DetailLayer {
    pub name: String,
    pub density: f32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub min_height: f32,
    pub max_height: f32,
    pub slope_range: (f32, f32),
    pub wind_sway: f32,
    pub color_variation: f32,
}

impl ProceduralTerrainDetail {
    pub fn new(view_distance: f32) -> Self {
        Self {
            detail_layers: Vec::new(),
            view_distance,
            density_map_size: 256,
        }
    }

    /// Generate detail instances for a terrain patch.
    pub fn generate_instances(
        &self,
        patch: &TerrainPatch,
        heightmap: &Heightmap,
        camera_pos: Vec3,
    ) -> Vec<DetailInstance> {
        let mut instances = Vec::new();

        for layer in &self.detail_layers {
            let grid_size = (patch.size / (1.0 / layer.density.max(0.1))).ceil() as u32;

            for gy in 0..grid_size {
                for gx in 0..grid_size {
                    // Position within patch
                    let lx = (gx as f32 + pseudo_random(gx, gy, 0) * 0.8) / grid_size as f32;
                    let ly = (gy as f32 + pseudo_random(gx, gy, 1) * 0.8) / grid_size as f32;

                    let wx = patch.world_x + lx * patch.size;
                    let wz = patch.world_z + ly * patch.size;

                    let dist = Vec2::new(wx - camera_pos.x, wz - camera_pos.z).length();
                    if dist > self.view_distance { continue; }

                    // Sample height
                    let u = (wx / (heightmap.width as f32 * heightmap.world_scale.x)).clamp(0.0, 1.0);
                    let v = (wz / (heightmap.height as f32 * heightmap.world_scale.z)).clamp(0.0, 1.0);
                    let h = heightmap.sample_bilinear(u, v) * heightmap.world_scale.y;

                    // Check height and slope constraints
                    if h < layer.min_height || h > layer.max_height { continue; }

                    let hm_x = (u * (heightmap.width - 1) as f32) as u32;
                    let hm_y = (v * (heightmap.height - 1) as f32) as u32;
                    let slope = heightmap.slope_at(hm_x.min(heightmap.width-1), hm_y.min(heightmap.height-1));
                    if slope < layer.slope_range.0 || slope > layer.slope_range.1 { continue; }

                    let scale = layer.min_scale + pseudo_random(gx, gy, 2) * (layer.max_scale - layer.min_scale);
                    let rotation = pseudo_random(gx, gy, 3) * 2.0 * PI;

                    // Distance-based fade
                    let fade = 1.0 - (dist / self.view_distance).clamp(0.0, 1.0);
                    let fade = fade * fade;

                    instances.push(DetailInstance {
                        position: Vec3::new(wx, h, wz),
                        scale,
                        rotation,
                        fade,
                        color_variation: pseudo_random(gx, gy, 4) * layer.color_variation,
                    });
                }
            }
        }

        instances
    }
}

#[derive(Debug, Clone)]
pub struct DetailInstance {
    pub position: Vec3,
    pub scale: f32,
    pub rotation: f32,
    pub fade: f32,
    pub color_variation: f32,
}

fn pseudo_random(x: u32, y: u32, seed: u32) -> f32 {
    let mut h = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(seed.wrapping_mul(1274126177));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h = h ^ (h >> 16);
    (h & 0xFFFF) as f32 / 65535.0
}

// ---------------------------------------------------------------------------
// Terrain rendering stats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct TerrainRenderStats {
    pub total_patches: u32,
    pub visible_patches: u32,
    pub total_triangles: u64,
    pub clipmap_updates: u32,
    pub virtual_texture_page_loads: u32,
    pub detail_instances: u32,
    pub snow_coverage_percent: f32,
    pub puddle_coverage_percent: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heightmap_bilinear() {
        let mut hm = Heightmap::new(4, 4, Vec3::new(1.0, 1.0, 1.0));
        hm.data = vec![0.0, 1.0, 0.0, 1.0,
                       1.0, 0.0, 1.0, 0.0,
                       0.0, 1.0, 0.0, 1.0,
                       1.0, 0.0, 1.0, 0.0];
        let center = hm.sample_bilinear(0.5, 0.5);
        assert!(center >= 0.0 && center <= 1.0);
    }

    #[test]
    fn test_clipmap_creation() {
        let hm = Heightmap::new(256, 256, Vec3::new(1.0, 50.0, 1.0));
        let cm = ClipmapTerrain::new(hm, 4, 64);
        assert_eq!(cm.levels.len(), 4);
    }

    #[test]
    fn test_virtual_texture() {
        let mut vt = VirtualTextureSystem::new(128, 8, 8, 1024, 1024);
        vt.request_page(0, 0, 0, 1.0);
        vt.update(1);
        assert_eq!(vt.lru_cache.len(), 1);
    }

    #[test]
    fn test_snow_accumulation() {
        let mut snow = SnowAccumulation::new(4, 4);
        snow.snowfall_intensity = 1.0;
        snow.temperature = -5.0;
        let hm = Heightmap::new(4, 4, Vec3::new(1.0, 1.0, 1.0));
        snow.update(1.0, &hm);
        assert!(snow.accumulation.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_material_blending() {
        let mut blender = TerrainMaterialBlender::new();
        blender.add_layer(TerrainLayer {
            name: "grass".into(),
            albedo_color: Vec3::new(0.2, 0.5, 0.1),
            min_height: 0.0,
            max_height: 100.0,
            ..Default::default()
        });
        blender.add_layer(TerrainLayer {
            name: "rock".into(),
            albedo_color: Vec3::new(0.4, 0.4, 0.4),
            min_height: 50.0,
            max_height: 200.0,
            ..Default::default()
        });

        let weights = blender.compute_weights(75.0, 0.3, &[0.5, 0.5]);
        assert_eq!(weights.len(), 2);
        let total: f32 = weights.iter().sum();
        assert!((total - 1.0).abs() < 0.01);
    }
}
