// engine/render/src/terrain_rendering_v2.rs - Enhanced terrain: virtual texturing,
// clipmap rendering, tessellation LOD, procedural detail, snow, puddles.
// (See terrain_rendering.rs for the original implementation; this is the v2 module.)

pub use super::terrain_rendering::*;

// Re-export from the existing terrain_rendering module and extend with v2 features.
// The full implementation is in the gen_all.py-generated terrain_rendering.rs file
// which this module wraps with additional v2 functionality.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct Vec2V2 { pub x: f32, pub y: f32 }
#[derive(Debug, Clone, Copy)]
pub struct Vec3V2 { pub x: f32, pub y: f32, pub z: f32 }

impl Vec3V2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn add(self, r: Self) -> Self { Self { x:self.x+r.x, y:self.y+r.y, z:self.z+r.z } }
    pub fn sub(self, r: Self) -> Self { Self { x:self.x-r.x, y:self.y-r.y, z:self.z-r.z } }
    pub fn scale(self, s: f32) -> Self { Self { x:self.x*s, y:self.y*s, z:self.z*s } }
    pub fn length(self) -> f32 { (self.x*self.x+self.y*self.y+self.z*self.z).sqrt() }
    pub fn normalize(self) -> Self { let l=self.length(); if l<1e-12{Self::ZERO}else{self.scale(1.0/l)} }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x+self.y*r.y+self.z*r.z }
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
}

/// Heightmap v2 with bilinear sampling and normal computation.
#[derive(Debug, Clone)]
pub struct HeightmapV2 {
    pub width: u32, pub height: u32, pub data: Vec<f32>,
    pub world_scale: Vec3V2,
}

impl HeightmapV2 {
    pub fn new(width: u32, height: u32, scale: Vec3V2) -> Self {
        Self { width, height, data: vec![0.0; (width * height) as usize], world_scale: scale }
    }

    pub fn sample_bilinear(&self, u: f32, v: f32) -> f32 {
        let fx = u * (self.width - 1) as f32;
        let fy = v * (self.height - 1) as f32;
        let x0 = (fx.floor() as u32).min(self.width - 2);
        let y0 = (fy.floor() as u32).min(self.height - 2);
        let tx = fx - x0 as f32; let ty = fy - y0 as f32;
        let h00 = self.data[(y0 * self.width + x0) as usize];
        let h10 = self.data[(y0 * self.width + x0 + 1) as usize];
        let h01 = self.data[((y0 + 1) * self.width + x0) as usize];
        let h11 = self.data[((y0 + 1) * self.width + x0 + 1) as usize];
        let h0 = h00 + tx * (h10 - h00); let h1 = h01 + tx * (h11 - h01);
        h0 + ty * (h1 - h0)
    }

    pub fn compute_normal(&self, x: u32, y: u32) -> Vec3V2 {
        let x0 = if x > 0 { x - 1 } else { 0 }; let x1 = (x + 1).min(self.width - 1);
        let y0 = if y > 0 { y - 1 } else { 0 }; let y1 = (y + 1).min(self.height - 1);
        let hl = self.data[(y * self.width + x0) as usize]; let hr = self.data[(y * self.width + x1) as usize];
        let hd = self.data[(y0 * self.width + x) as usize]; let hu = self.data[(y1 * self.width + x) as usize];
        let dx = (hr - hl) * self.world_scale.y / (2.0 * self.world_scale.x);
        let dz = (hu - hd) * self.world_scale.y / (2.0 * self.world_scale.z);
        Vec3V2::new(-dx, 1.0, -dz).normalize()
    }

    pub fn slope_at(&self, x: u32, y: u32) -> f32 {
        let n = self.compute_normal(x, y);
        n.dot(Vec3V2::new(0.0, 1.0, 0.0)).acos()
    }
}

/// Clipmap level for terrain LOD.
#[derive(Debug, Clone)]
pub struct ClipmapLevelV2 { pub level: u32, pub size: u32, pub texel_size: f32, pub center: Vec2V2, pub dirty: bool }

/// Virtual texture page table entry.
#[derive(Debug, Clone, Copy)]
pub struct VTPageEntry { pub physical_x: u16, pub physical_y: u16, pub mip: u8, pub valid: bool }
impl Default for VTPageEntry { fn default() -> Self { Self { physical_x: 0, physical_y: 0, mip: 0, valid: false } } }

/// Virtual texture system for terrain materials.
pub struct VirtualTextureV2 {
    pub page_size: u32, pub physical_x: u32, pub physical_y: u32,
    pub page_table: Vec<VTPageEntry>, pub pt_width: u32, pub pt_height: u32,
    pub frame_budget: u32, pub pending: Vec<(u32, u32, u32, f32)>,
}

impl VirtualTextureV2 {
    pub fn new(page_size: u32, phys_x: u32, phys_y: u32, virt_w: u32, virt_h: u32) -> Self {
        let pt_w = virt_w / page_size; let pt_h = virt_h / page_size;
        Self { page_size, physical_x: phys_x, physical_y: phys_y,
            page_table: vec![VTPageEntry::default(); (pt_w * pt_h) as usize],
            pt_width: pt_w, pt_height: pt_h, frame_budget: 8, pending: Vec::new() }
    }
    pub fn request_page(&mut self, vx: u32, vy: u32, mip: u32, priority: f32) {
        let idx = (vy * self.pt_width + vx) as usize;
        if idx < self.page_table.len() && self.page_table[idx].valid { return; }
        self.pending.push((vx, vy, mip, priority));
    }
}

/// Snow accumulation on terrain.
pub struct SnowAccumulationV2 {
    pub width: u32, pub height: u32, pub accumulation: Vec<f32>,
    pub max_depth: f32, pub rate: f32, pub melt_rate: f32,
    pub slope_threshold: f32, pub temperature: f32, pub intensity: f32,
}

impl SnowAccumulationV2 {
    pub fn new(w: u32, h: u32) -> Self {
        Self { width: w, height: h, accumulation: vec![0.0; (w*h) as usize],
            max_depth: 1.0, rate: 0.01, melt_rate: 0.005, slope_threshold: 0.8,
            temperature: -5.0, intensity: 0.5 }
    }
    pub fn update(&mut self, dt: f32, heightmap: &HeightmapV2) {
        for y in 0..self.height { for x in 0..self.width {
            let idx = (y * self.width + x) as usize;
            let hx = (x as f32 / self.width as f32 * heightmap.width as f32) as u32;
            let hy = (y as f32 / self.height as f32 * heightmap.height as f32) as u32;
            let slope = heightmap.slope_at(hx.min(heightmap.width-1), hy.min(heightmap.height-1));
            if self.intensity > 0.0 && slope < self.slope_threshold && self.temperature < 0.0 {
                let sf = 1.0 - (slope / self.slope_threshold);
                self.accumulation[idx] += self.rate * self.intensity * sf * dt;
            }
            if self.temperature > 0.0 { self.accumulation[idx] -= self.melt_rate * self.temperature * dt; }
            self.accumulation[idx] = self.accumulation[idx].clamp(0.0, self.max_depth);
        }}
    }
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let x = (u * (self.width-1) as f32).round() as u32;
        let y = (v * (self.height-1) as f32).round() as u32;
        self.accumulation[(y.min(self.height-1) * self.width + x.min(self.width-1)) as usize]
    }
}

/// Puddle system for rain effects.
pub struct PuddleSystemV2 {
    pub width: u32, pub height: u32, pub water_depth: Vec<f32>,
    pub rain_intensity: f32, pub drain_rate: f32,
}

impl PuddleSystemV2 {
    pub fn new(w: u32, h: u32) -> Self {
        Self { width: w, height: h, water_depth: vec![0.0; (w*h) as usize], rain_intensity: 0.0, drain_rate: 0.001 }
    }
    pub fn update(&mut self, dt: f32) {
        for d in &mut self.water_depth {
            if self.rain_intensity > 0.0 { *d += self.rain_intensity * 0.001 * dt; }
            *d -= self.drain_rate * dt;
            *d = d.clamp(0.0, 0.05);
        }
    }
}

/// Terrain material layer.
#[derive(Debug, Clone)]
pub struct TerrainLayerV2 {
    pub name: String, pub albedo: Vec3V2, pub roughness: f32,
    pub min_height: f32, pub max_height: f32, pub min_slope: f32, pub max_slope: f32,
}

/// Terrain material blending.
pub struct TerrainBlenderV2 { pub layers: Vec<TerrainLayerV2> }

impl TerrainBlenderV2 {
    pub fn new() -> Self { Self { layers: Vec::new() } }
    pub fn add_layer(&mut self, layer: TerrainLayerV2) { self.layers.push(layer); }
    pub fn compute_weights(&self, height: f32, slope: f32) -> Vec<f32> {
        let mut weights: Vec<f32> = self.layers.iter().map(|l| {
            if height < l.min_height || height > l.max_height { return 0.0; }
            if slope < l.min_slope || slope > l.max_slope { return 0.0; }
            1.0
        }).collect();
        let total: f32 = weights.iter().sum();
        if total > 0.0 { for w in &mut weights { *w /= total; } }
        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_heightmap() {
        let hm = HeightmapV2::new(4, 4, Vec3V2::new(1.0, 1.0, 1.0));
        let h = hm.sample_bilinear(0.5, 0.5);
        assert!(h >= 0.0);
    }
    #[test] fn test_snow() {
        let mut snow = SnowAccumulationV2::new(4, 4);
        snow.intensity = 1.0;
        let hm = HeightmapV2::new(4, 4, Vec3V2::new(1.0, 1.0, 1.0));
        snow.update(1.0, &hm);
        assert!(snow.accumulation.iter().any(|&v| v > 0.0));
    }
    #[test] fn test_vt() {
        let mut vt = VirtualTextureV2::new(128, 8, 8, 1024, 1024);
        vt.request_page(0, 0, 0, 1.0);
        assert_eq!(vt.pending.len(), 1);
    }
}
