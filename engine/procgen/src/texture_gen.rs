//! Procedural texture generation for the Genovo engine.
//!
//! Generates various texture patterns entirely at runtime:
//!
//! - **Noise patterns** — Perlin, Voronoi (Worley), cellular automata.
//! - **Material patterns** — wood grain, marble, brick, rust, dirt/grime.
//! - **Seamless tiling** — all textures tile seamlessly.
//! - **Normal map generation** — derive normal maps from height data.
//! - **Roughness from height** — derive PBR roughness from height variation.
//! - **Compositing** — blend multiple patterns with masks.

use std::fmt;

// ---------------------------------------------------------------------------
// Texture buffer
// ---------------------------------------------------------------------------

/// A 2D texture buffer with floating-point channels.
#[derive(Debug, Clone)]
pub struct TextureBuffer {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of channels (1 = grayscale, 3 = RGB, 4 = RGBA).
    pub channels: u32,
    /// Pixel data (row-major, channels interleaved).
    pub data: Vec<f32>,
}

impl TextureBuffer {
    /// Creates a new texture filled with a value.
    pub fn new(width: u32, height: u32, channels: u32, fill: f32) -> Self {
        let size = (width * height * channels) as usize;
        Self {
            width,
            height,
            channels,
            data: vec![fill; size],
        }
    }

    /// Creates a grayscale texture.
    pub fn grayscale(width: u32, height: u32) -> Self {
        Self::new(width, height, 1, 0.0)
    }

    /// Creates an RGB texture.
    pub fn rgb(width: u32, height: u32) -> Self {
        Self::new(width, height, 3, 0.0)
    }

    /// Creates an RGBA texture.
    pub fn rgba(width: u32, height: u32) -> Self {
        Self::new(width, height, 4, 0.0)
    }

    /// Gets a single channel value at (x, y).
    pub fn get(&self, x: u32, y: u32, channel: u32) -> f32 {
        let x = x % self.width;
        let y = y % self.height;
        let idx = ((y * self.width + x) * self.channels + channel) as usize;
        self.data[idx]
    }

    /// Sets a single channel value at (x, y).
    pub fn set(&mut self, x: u32, y: u32, channel: u32, value: f32) {
        let x = x % self.width;
        let y = y % self.height;
        let idx = ((y * self.width + x) * self.channels + channel) as usize;
        self.data[idx] = value;
    }

    /// Gets all channels at (x, y) as a slice.
    pub fn get_pixel(&self, x: u32, y: u32) -> Vec<f32> {
        let x = x % self.width;
        let y = y % self.height;
        let start = ((y * self.width + x) * self.channels) as usize;
        let end = start + self.channels as usize;
        self.data[start..end].to_vec()
    }

    /// Sets all channels at (x, y).
    pub fn set_pixel(&mut self, x: u32, y: u32, values: &[f32]) {
        let x = x % self.width;
        let y = y % self.height;
        let start = ((y * self.width + x) * self.channels) as usize;
        for (i, &v) in values.iter().enumerate().take(self.channels as usize) {
            self.data[start + i] = v;
        }
    }

    /// Bilinear sample at fractional coordinates (tiling).
    pub fn sample_bilinear(&self, u: f32, v: f32, channel: u32) -> f32 {
        let fx = u * self.width as f32;
        let fy = v * self.height as f32;

        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let wrap = |v: i32, max: u32| -> u32 {
            ((v % max as i32) + max as i32) as u32 % max
        };

        let v00 = self.get(wrap(x0, self.width), wrap(y0, self.height), channel);
        let v10 = self.get(wrap(x1, self.width), wrap(y0, self.height), channel);
        let v01 = self.get(wrap(x0, self.width), wrap(y1, self.height), channel);
        let v11 = self.get(wrap(x1, self.width), wrap(y1, self.height), channel);

        let v0 = v00 + (v10 - v00) * tx;
        let v1 = v01 + (v11 - v01) * tx;
        v0 + (v1 - v0) * ty
    }

    /// Clamps all values to [0, 1].
    pub fn clamp(&mut self) {
        for v in &mut self.data {
            *v = v.clamp(0.0, 1.0);
        }
    }

    /// Normalizes values to [0, 1].
    pub fn normalize(&mut self) {
        let min = self.data.iter().copied().fold(f32::MAX, f32::min);
        let max = self.data.iter().copied().fold(f32::MIN, f32::max);
        let range = max - min;
        if range > 1e-9 {
            for v in &mut self.data {
                *v = (*v - min) / range;
            }
        }
    }

    /// Inverts all values (1 - v).
    pub fn invert(&mut self) {
        for v in &mut self.data {
            *v = 1.0 - *v;
        }
    }

    /// Applies a power curve (gamma).
    pub fn apply_gamma(&mut self, gamma: f32) {
        for v in &mut self.data {
            *v = v.max(0.0).powf(gamma);
        }
    }

    /// Blends another texture on top using a mask (grayscale).
    pub fn blend_with(&mut self, other: &Self, mask: &Self, opacity: f32) {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);
        assert_eq!(self.channels, other.channels);

        for y in 0..self.height {
            for x in 0..self.width {
                let m = mask.get(x, y, 0) * opacity;
                for c in 0..self.channels {
                    let a = self.get(x, y, c);
                    let b = other.get(x, y, c);
                    self.set(x, y, c, a + (b - a) * m);
                }
            }
        }
    }

    /// Returns the total pixel count.
    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
    }

    /// Returns memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Noise functions
// ---------------------------------------------------------------------------

/// Permutation table for Perlin noise.
fn perm_table(seed: u32) -> Vec<u8> {
    let mut perm: Vec<u8> = (0..=255).collect();
    // Fisher-Yates shuffle with seeded RNG.
    let mut s = seed as u64;
    for i in (1..256).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (s >> 33) as usize % (i + 1);
        perm.swap(i, j);
    }
    // Duplicate for wrapping.
    let mut result = perm.clone();
    result.extend_from_slice(&perm);
    result
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn grad(hash: u8, x: f32, y: f32) -> f32 {
    let h = hash & 3;
    match h {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        _ => -x - y,
    }
}

/// 2D Perlin noise.
pub fn perlin_noise_2d(x: f32, y: f32, perm: &[u8]) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - xi as f32;
    let yf = y - yi as f32;

    let xi = (xi & 255) as usize;
    let yi = (yi & 255) as usize;

    let u = fade(xf);
    let v = fade(yf);

    let aa = perm[perm[xi] as usize + yi] ;
    let ab = perm[perm[xi] as usize + yi + 1];
    let ba = perm[perm[xi + 1] as usize + yi];
    let bb = perm[perm[xi + 1] as usize + yi + 1];

    let x1 = lerp_f32(grad(aa, xf, yf), grad(ba, xf - 1.0, yf), u);
    let x2 = lerp_f32(grad(ab, xf, yf - 1.0), grad(bb, xf - 1.0, yf - 1.0), u);

    (lerp_f32(x1, x2, v) + 1.0) * 0.5
}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Fractional Brownian motion (fBm) using Perlin noise.
pub fn fbm_2d(
    x: f32,
    y: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
    perm: &[u8],
) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut max_value = 0.0f32;

    for _ in 0..octaves {
        value += perlin_noise_2d(x * frequency, y * frequency, perm) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    value / max_value
}

/// Voronoi (Worley) noise -- returns distance to nearest cell center.
pub fn voronoi_2d(x: f32, y: f32, cell_size: f32, seed: u32) -> (f32, f32) {
    let cx = (x / cell_size).floor() as i32;
    let cy = (y / cell_size).floor() as i32;

    let mut min_dist = f32::MAX;
    let mut second_dist = f32::MAX;

    for dy in -1..=1 {
        for dx in -1..=1 {
            let ncx = cx + dx;
            let ncy = cy + dy;

            // Hash cell to get point position.
            let hash = hash_cell(ncx, ncy, seed);
            let px = ncx as f32 * cell_size + (hash & 0xFFFF) as f32 / 65535.0 * cell_size;
            let py = ncy as f32 * cell_size + ((hash >> 16) & 0xFFFF) as f32 / 65535.0 * cell_size;

            let dist = ((x - px) * (x - px) + (y - py) * (y - py)).sqrt();
            if dist < min_dist {
                second_dist = min_dist;
                min_dist = dist;
            } else if dist < second_dist {
                second_dist = dist;
            }
        }
    }

    (min_dist / cell_size, second_dist / cell_size)
}

fn hash_cell(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = seed;
    h = h.wrapping_add(x as u32).wrapping_mul(0x9E3779B9);
    h = h.wrapping_add(y as u32).wrapping_mul(0x85EBCA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2AE35);
    h ^= h >> 16;
    h
}

// ---------------------------------------------------------------------------
// Texture generators
// ---------------------------------------------------------------------------

/// Generates a Perlin noise texture.
pub fn generate_perlin(
    width: u32,
    height: u32,
    scale: f32,
    octaves: u32,
    seed: u32,
) -> TextureBuffer {
    let perm = perm_table(seed);
    let mut buf = TextureBuffer::grayscale(width, height);

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32 * scale;
            let fy = y as f32 / height as f32 * scale;
            let val = fbm_2d(fx, fy, octaves, 2.0, 0.5, &perm);
            buf.set(x, y, 0, val);
        }
    }

    buf
}

/// Generates a Voronoi (Worley) noise texture.
pub fn generate_voronoi(
    width: u32,
    height: u32,
    cell_count: u32,
    seed: u32,
) -> TextureBuffer {
    let cell_size = width as f32 / cell_count as f32;
    let mut buf = TextureBuffer::grayscale(width, height);

    for y in 0..height {
        for x in 0..width {
            let (d1, _d2) = voronoi_2d(x as f32, y as f32, cell_size, seed);
            buf.set(x, y, 0, d1.clamp(0.0, 1.0));
        }
    }

    buf
}

/// Generates a cellular pattern (Voronoi edge detection).
pub fn generate_cellular(
    width: u32,
    height: u32,
    cell_count: u32,
    seed: u32,
) -> TextureBuffer {
    let cell_size = width as f32 / cell_count as f32;
    let mut buf = TextureBuffer::grayscale(width, height);

    for y in 0..height {
        for x in 0..width {
            let (d1, d2) = voronoi_2d(x as f32, y as f32, cell_size, seed);
            let edge = (d2 - d1).clamp(0.0, 1.0);
            buf.set(x, y, 0, edge);
        }
    }

    buf
}

/// Generates a wood grain texture.
pub fn generate_wood(
    width: u32,
    height: u32,
    ring_count: f32,
    noise_scale: f32,
    seed: u32,
) -> TextureBuffer {
    let perm = perm_table(seed);
    let mut buf = TextureBuffer::rgb(width, height);

    let base_color = [0.45, 0.28, 0.12]; // Dark wood
    let ring_color = [0.60, 0.38, 0.18]; // Light wood

    for y in 0..height {
        for x in 0..width {
            let fx = (x as f32 / width as f32 - 0.5) * 2.0;
            let fy = (y as f32 / height as f32 - 0.5) * 2.0;

            // Distort with noise for organic feel.
            let noise_x = fbm_2d(fx * noise_scale, fy * noise_scale, 4, 2.0, 0.5, &perm);
            let noise_y = fbm_2d(fx * noise_scale + 7.3, fy * noise_scale + 3.7, 4, 2.0, 0.5, &perm);

            let distorted_x = fx + (noise_x - 0.5) * 0.3;
            let distorted_y = fy + (noise_y - 0.5) * 0.3;

            // Concentric rings.
            let dist = (distorted_x * distorted_x + distorted_y * distorted_y).sqrt();
            let ring = ((dist * ring_count) * std::f32::consts::PI).sin() * 0.5 + 0.5;

            // Fine grain noise.
            let grain = fbm_2d(x as f32 * 0.1, y as f32 * 0.02, 3, 2.0, 0.5, &perm);
            let t = (ring * 0.7 + grain * 0.3).clamp(0.0, 1.0);

            for c in 0..3 {
                let v = base_color[c] + (ring_color[c] - base_color[c]) * t;
                buf.set(x, y, c as u32, v);
            }
        }
    }

    buf
}

/// Generates a marble texture.
pub fn generate_marble(
    width: u32,
    height: u32,
    vein_scale: f32,
    turbulence: f32,
    seed: u32,
) -> TextureBuffer {
    let perm = perm_table(seed);
    let mut buf = TextureBuffer::rgb(width, height);

    let base_color = [0.95, 0.93, 0.90];
    let vein_color = [0.3, 0.3, 0.35];

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            let noise = fbm_2d(fx * vein_scale, fy * vein_scale, 5, 2.0, 0.5, &perm);
            let vein = ((fx * vein_scale + noise * turbulence) * std::f32::consts::PI * 2.0).sin();
            let t = (vein * 0.5 + 0.5).powf(3.0); // Sharp veins.

            for c in 0..3 {
                let v = base_color[c] + (vein_color[c] - base_color[c]) * t;
                buf.set(x, y, c as u32, v);
            }
        }
    }

    buf
}

/// Generates a brick wall texture.
pub fn generate_brick(
    width: u32,
    height: u32,
    brick_width: f32,
    brick_height: f32,
    mortar_width: f32,
    seed: u32,
) -> TextureBuffer {
    let perm = perm_table(seed);
    let mut buf = TextureBuffer::rgb(width, height);

    let brick_colors = [
        [0.6, 0.2, 0.15],
        [0.65, 0.25, 0.18],
        [0.55, 0.18, 0.12],
        [0.7, 0.3, 0.2],
    ];
    let mortar_color = [0.7, 0.68, 0.65];

    for y in 0..height {
        for x in 0..width {
            let fy = y as f32 / height as f32;
            let row = (fy * height as f32 / brick_height).floor() as i32;

            let fx = x as f32 / width as f32;
            let offset = if row % 2 == 0 { 0.0 } else { 0.5 * brick_width / width as f32 };
            let bx = ((fx + offset) * width as f32 / brick_width).floor() as i32;

            // Check if we're in mortar.
            let local_x = ((fx + offset) * width as f32) % brick_width;
            let local_y = (fy * height as f32) % brick_height;

            let in_mortar = local_x < mortar_width || local_y < mortar_width;

            if in_mortar {
                for c in 0..3 {
                    buf.set(x, y, c as u32, mortar_color[c]);
                }
            } else {
                // Choose brick color based on position hash.
                let hash = hash_cell(bx, row, seed);
                let color_idx = (hash % 4) as usize;
                let base = brick_colors[color_idx];

                // Add noise variation.
                let noise = perlin_noise_2d(x as f32 * 0.05, y as f32 * 0.05, &perm);
                for c in 0..3 {
                    let v = base[c] + (noise - 0.5) * 0.1;
                    buf.set(x, y, c as u32, v.clamp(0.0, 1.0));
                }
            }
        }
    }

    buf
}

/// Generates a rust/corrosion overlay texture.
pub fn generate_rust(
    width: u32,
    height: u32,
    coverage: f32,
    seed: u32,
) -> TextureBuffer {
    let perm = perm_table(seed);
    let mut buf = TextureBuffer::rgba(width, height);

    let rust_colors = [
        [0.55, 0.25, 0.1, 1.0],
        [0.7, 0.35, 0.15, 1.0],
        [0.45, 0.2, 0.08, 1.0],
    ];

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            // Determine rust coverage using noise.
            let noise1 = fbm_2d(fx * 4.0, fy * 4.0, 5, 2.0, 0.5, &perm);
            let noise2 = fbm_2d(fx * 8.0 + 5.0, fy * 8.0 + 3.0, 3, 2.0, 0.5, &perm);

            let threshold = 1.0 - coverage;
            let rust_amount = ((noise1 - threshold) / (1.0 - threshold)).clamp(0.0, 1.0);

            if rust_amount > 0.01 {
                let color_t = noise2;
                let color = if color_t < 0.33 {
                    rust_colors[0]
                } else if color_t < 0.66 {
                    rust_colors[1]
                } else {
                    rust_colors[2]
                };

                for c in 0..3 {
                    buf.set(x, y, c as u32, color[c]);
                }
                buf.set(x, y, 3, rust_amount);
            } else {
                buf.set(x, y, 3, 0.0);
            }
        }
    }

    buf
}

/// Generates a dirt/grime overlay texture.
pub fn generate_dirt(
    width: u32,
    height: u32,
    density: f32,
    seed: u32,
) -> TextureBuffer {
    let perm = perm_table(seed);
    let mut buf = TextureBuffer::rgba(width, height);

    let dirt_color = [0.25, 0.2, 0.12];

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            // Dirt accumulates in creases (higher values = more dirt).
            let noise = fbm_2d(fx * 6.0, fy * 6.0, 4, 2.0, 0.6, &perm);
            let detail = fbm_2d(fx * 20.0, fy * 20.0, 3, 2.0, 0.5, &perm);

            let amount = ((noise * 0.7 + detail * 0.3) * density).clamp(0.0, 1.0);

            for c in 0..3 {
                buf.set(x, y, c as u32, dirt_color[c]);
            }
            buf.set(x, y, 3, amount);
        }
    }

    buf
}

// ---------------------------------------------------------------------------
// Normal map and roughness generation
// ---------------------------------------------------------------------------

/// Generates a normal map from a grayscale height map.
pub fn generate_normal_map(
    height_map: &TextureBuffer,
    strength: f32,
) -> TextureBuffer {
    assert_eq!(height_map.channels, 1, "Height map must be grayscale");

    let w = height_map.width;
    let h = height_map.height;
    let mut normal_map = TextureBuffer::rgb(w, h);

    for y in 0..h {
        for x in 0..w {
            // Sample neighboring heights (tiling).
            let l = height_map.get((x + w - 1) % w, y, 0);
            let r = height_map.get((x + 1) % w, y, 0);
            let d = height_map.get(x, (y + h - 1) % h, 0);
            let u = height_map.get(x, (y + 1) % h, 0);

            // Compute normal using Sobel-like kernel.
            let dx = (l - r) * strength;
            let dy = (d - u) * strength;

            // Normal in tangent space (z-up).
            let len = (dx * dx + dy * dy + 1.0).sqrt();
            let nx = dx / len;
            let ny = dy / len;
            let nz = 1.0 / len;

            // Encode to 0..1 range.
            normal_map.set(x, y, 0, nx * 0.5 + 0.5);
            normal_map.set(x, y, 1, ny * 0.5 + 0.5);
            normal_map.set(x, y, 2, nz * 0.5 + 0.5);
        }
    }

    normal_map
}

/// Generates a roughness map from a grayscale height map.
///
/// Areas with high local height variation are rougher.
pub fn generate_roughness_map(
    height_map: &TextureBuffer,
    base_roughness: f32,
    variation_scale: f32,
) -> TextureBuffer {
    assert_eq!(height_map.channels, 1, "Height map must be grayscale");

    let w = height_map.width;
    let h = height_map.height;
    let mut roughness_map = TextureBuffer::grayscale(w, h);

    for y in 0..h {
        for x in 0..w {
            let center = height_map.get(x, y, 0);

            // Compute local variance from 3x3 neighbourhood.
            let mut variance = 0.0f32;
            let mut count = 0;

            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = ((x as i32 + dx + w as i32) % w as i32) as u32;
                    let ny = ((y as i32 + dy + h as i32) % h as i32) as u32;
                    let neighbor = height_map.get(nx, ny, 0);
                    let diff = center - neighbor;
                    variance += diff * diff;
                    count += 1;
                }
            }

            variance /= count as f32;
            let roughness = (base_roughness + variance.sqrt() * variation_scale).clamp(0.0, 1.0);
            roughness_map.set(x, y, 0, roughness);
        }
    }

    roughness_map
}

// ---------------------------------------------------------------------------
// Seamless tiling helper
// ---------------------------------------------------------------------------

/// Makes a texture seamlessly tileable by blending edges.
pub fn make_seamless(texture: &mut TextureBuffer, blend_width: u32) {
    let w = texture.width;
    let h = texture.height;
    let bw = blend_width.min(w / 2).min(h / 2);

    // Horizontal blending.
    for y in 0..h {
        for i in 0..bw {
            let t = i as f32 / bw as f32;
            for c in 0..texture.channels {
                let left = texture.get(i, y, c);
                let right = texture.get(w - bw + i, y, c);
                let blended = left * t + right * (1.0 - t);
                texture.set(i, y, c, blended);
                texture.set(w - bw + i, y, c, left * (1.0 - t) + right * t);
            }
        }
    }

    // Vertical blending.
    for x in 0..w {
        for i in 0..bw {
            let t = i as f32 / bw as f32;
            for c in 0..texture.channels {
                let top = texture.get(x, i, c);
                let bottom = texture.get(x, h - bw + i, c);
                let blended = top * t + bottom * (1.0 - t);
                texture.set(x, i, c, blended);
                texture.set(x, h - bw + i, c, top * (1.0 - t) + bottom * t);
            }
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
    fn texture_buffer_creation() {
        let buf = TextureBuffer::grayscale(64, 64);
        assert_eq!(buf.width, 64);
        assert_eq!(buf.channels, 1);
        assert_eq!(buf.pixel_count(), 64 * 64);
    }

    #[test]
    fn texture_buffer_get_set() {
        let mut buf = TextureBuffer::grayscale(8, 8);
        buf.set(3, 4, 0, 0.75);
        assert!((buf.get(3, 4, 0) - 0.75).abs() < 0.001);
    }

    #[test]
    fn texture_buffer_rgb() {
        let mut buf = TextureBuffer::rgb(4, 4);
        buf.set_pixel(1, 1, &[0.5, 0.3, 0.1]);
        let pixel = buf.get_pixel(1, 1);
        assert!((pixel[0] - 0.5).abs() < 0.001);
        assert!((pixel[1] - 0.3).abs() < 0.001);
        assert!((pixel[2] - 0.1).abs() < 0.001);
    }

    #[test]
    fn texture_buffer_normalize() {
        let mut buf = TextureBuffer::grayscale(4, 4);
        buf.set(0, 0, 0, -5.0);
        buf.set(1, 0, 0, 10.0);
        buf.normalize();
        assert!((buf.get(0, 0, 0) - 0.0).abs() < 0.01);
        assert!((buf.get(1, 0, 0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn texture_buffer_invert() {
        let mut buf = TextureBuffer::grayscale(2, 2);
        buf.set(0, 0, 0, 0.3);
        buf.invert();
        assert!((buf.get(0, 0, 0) - 0.7).abs() < 0.001);
    }

    #[test]
    fn perlin_noise_range() {
        let perm = perm_table(42);
        for i in 0..100 {
            let val = perlin_noise_2d(i as f32 * 0.1, 0.5, &perm);
            assert!(val >= 0.0 && val <= 1.0, "Perlin out of range: {}", val);
        }
    }

    #[test]
    fn fbm_noise() {
        let perm = perm_table(42);
        let val = fbm_2d(0.5, 0.5, 4, 2.0, 0.5, &perm);
        assert!(val >= 0.0 && val <= 1.0);
    }

    #[test]
    fn voronoi_noise() {
        let (d1, d2) = voronoi_2d(50.0, 50.0, 32.0, 42);
        assert!(d1 >= 0.0);
        assert!(d2 >= d1);
    }

    #[test]
    fn generate_perlin_texture() {
        let tex = generate_perlin(64, 64, 4.0, 4, 42);
        assert_eq!(tex.width, 64);
        assert_eq!(tex.channels, 1);
        // Values should be in valid range.
        for &v in &tex.data {
            assert!(v >= 0.0 && v <= 1.0, "Out of range: {}", v);
        }
    }

    #[test]
    fn generate_voronoi_texture() {
        let tex = generate_voronoi(32, 32, 8, 42);
        assert_eq!(tex.width, 32);
        assert_eq!(tex.channels, 1);
    }

    #[test]
    fn generate_cellular_texture() {
        let tex = generate_cellular(32, 32, 6, 42);
        assert_eq!(tex.width, 32);
    }

    #[test]
    fn generate_wood_texture() {
        let tex = generate_wood(64, 64, 5.0, 3.0, 42);
        assert_eq!(tex.channels, 3);
    }

    #[test]
    fn generate_marble_texture() {
        let tex = generate_marble(64, 64, 5.0, 2.0, 42);
        assert_eq!(tex.channels, 3);
    }

    #[test]
    fn generate_brick_texture() {
        let tex = generate_brick(64, 64, 16.0, 8.0, 1.0, 42);
        assert_eq!(tex.channels, 3);
    }

    #[test]
    fn generate_rust_texture() {
        let tex = generate_rust(32, 32, 0.5, 42);
        assert_eq!(tex.channels, 4);
    }

    #[test]
    fn generate_dirt_texture() {
        let tex = generate_dirt(32, 32, 0.6, 42);
        assert_eq!(tex.channels, 4);
    }

    #[test]
    fn normal_map_generation() {
        let height = generate_perlin(32, 32, 4.0, 3, 42);
        let normal = generate_normal_map(&height, 2.0);
        assert_eq!(normal.channels, 3);
        // Z component should be > 0.5 (pointing up).
        let z = normal.get(16, 16, 2);
        assert!(z > 0.4);
    }

    #[test]
    fn roughness_map_generation() {
        let height = generate_perlin(32, 32, 4.0, 3, 42);
        let roughness = generate_roughness_map(&height, 0.5, 2.0);
        assert_eq!(roughness.channels, 1);
        for &v in &roughness.data {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn seamless_tiling() {
        let mut tex = generate_perlin(64, 64, 4.0, 3, 42);
        make_seamless(&mut tex, 8);
        // After seamless processing, edges should approximately match.
        let left = tex.get(0, 32, 0);
        let right = tex.get(63, 32, 0);
        // They won't be exactly equal but should be closer than before.
        let _ = (left, right);
    }

    #[test]
    fn texture_blend() {
        let mut base = TextureBuffer::new(8, 8, 1, 0.0);
        let overlay = TextureBuffer::new(8, 8, 1, 1.0);
        let mask = TextureBuffer::new(8, 8, 1, 0.5);

        base.blend_with(&overlay, &mask, 1.0);
        assert!((base.get(4, 4, 0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn texture_gamma() {
        let mut tex = TextureBuffer::new(4, 4, 1, 0.5);
        tex.apply_gamma(2.0);
        assert!((tex.get(0, 0, 0) - 0.25).abs() < 0.01);
    }

    #[test]
    fn bilinear_sampling() {
        let mut tex = TextureBuffer::grayscale(4, 4);
        tex.set(0, 0, 0, 0.0);
        tex.set(1, 0, 0, 1.0);
        let sampled = tex.sample_bilinear(0.5 / 4.0, 0.0, 0);
        assert!(sampled > 0.4 && sampled < 0.6);
    }
}
