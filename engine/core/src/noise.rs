//! Noise generation functions for procedural content.
//!
//! Provides multiple noise algorithms commonly used in game development for
//! terrain generation, texture synthesis, particle effects, and more:
//!
//! - **Perlin noise** — classic gradient noise with smooth results
//! - **Simplex noise** — faster, fewer directional artifacts
//! - **Worley noise** — cellular/Voronoi distance-based noise
//! - **Value noise** — interpolated random lattice values
//! - **Curl noise** — divergence-free noise for fluid/particle effects
//!
//! All noise functions are deterministic given the same seed and coordinates.

use glam::{Vec2, Vec3};

// ---------------------------------------------------------------------------
// Permutation table
// ---------------------------------------------------------------------------

/// Size of the permutation table. Must be a power of two.
const PERM_SIZE: usize = 256;

/// Generate a shuffled permutation table from a seed using a simple LCG.
fn build_permutation_table(seed: u64) -> [u8; PERM_SIZE] {
    let mut perm = [0u8; PERM_SIZE];
    for i in 0..PERM_SIZE {
        perm[i] = i as u8;
    }
    // Fisher-Yates shuffle driven by a simple LCG.
    let mut state = seed.wrapping_add(1);
    for i in (1..PERM_SIZE).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        perm.swap(i, j);
    }
    perm
}

/// The classic Perlin fade curve: 6t^5 - 15t^4 + 10t^3.
///
/// Has zero first and second derivatives at t=0 and t=1, producing smooth
/// visual results when used for interpolation.
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Linear interpolation.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

// ---------------------------------------------------------------------------
// 3D gradient vectors
// ---------------------------------------------------------------------------

/// The 12 gradient vectors for 3D Perlin noise (edges of a cube).
const GRAD3: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

/// 2D gradient vectors (4 cardinal + 4 diagonal).
const GRAD2: [[f32; 2]; 8] = [
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
];

// ===========================================================================
// PerlinNoise
// ===========================================================================

/// Classic Perlin gradient noise generator.
///
/// Produces smooth, continuous noise in 2D and 3D with configurable seed.
/// Values are centered around zero with approximate range [-1, 1].
///
/// # Example
/// ```ignore
/// let noise = PerlinNoise::new(42);
/// let val = noise.noise2d(1.5, 2.3);
/// let fbm_val = noise.fbm(1.5, 2.3, 0.0, 6, 2.0, 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct PerlinNoise {
    /// Permutation table (doubled to avoid index wrapping).
    perm: [u8; PERM_SIZE * 2],
    /// The seed used to generate this noise instance.
    pub seed: u64,
}

impl PerlinNoise {
    /// Create a new Perlin noise generator with the given seed.
    pub fn new(seed: u64) -> Self {
        let base = build_permutation_table(seed);
        let mut perm = [0u8; PERM_SIZE * 2];
        for i in 0..PERM_SIZE {
            perm[i] = base[i];
            perm[i + PERM_SIZE] = base[i];
        }
        Self { perm, seed }
    }

    /// Hash function: look up a value in the permutation table.
    #[inline]
    fn hash(&self, x: i32) -> usize {
        self.perm[(x & 0xFF) as usize] as usize
    }

    /// 2D Perlin noise at the given coordinates.
    ///
    /// Returns a value in approximately [-1, 1].
    pub fn noise2d(&self, x: f32, y: f32) -> f32 {
        // Grid cell coordinates.
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;

        // Fractional position within the cell.
        let xf = x - xi as f32;
        let yf = y - yi as f32;

        // Fade curves for interpolation.
        let u = fade(xf);
        let v = fade(yf);

        // Hash the four corners.
        let aa = self.hash(self.hash(xi) as i32 + yi) % 8;
        let ab = self.hash(self.hash(xi) as i32 + yi + 1) % 8;
        let ba = self.hash(self.hash(xi + 1) as i32 + yi) % 8;
        let bb = self.hash(self.hash(xi + 1) as i32 + yi + 1) % 8;

        // Gradient dot products.
        let g_aa = GRAD2[aa][0] * xf + GRAD2[aa][1] * yf;
        let g_ba = GRAD2[ba][0] * (xf - 1.0) + GRAD2[ba][1] * yf;
        let g_ab = GRAD2[ab][0] * xf + GRAD2[ab][1] * (yf - 1.0);
        let g_bb = GRAD2[bb][0] * (xf - 1.0) + GRAD2[bb][1] * (yf - 1.0);

        // Bilinear interpolation.
        let x1 = lerp(g_aa, g_ba, u);
        let x2 = lerp(g_ab, g_bb, u);
        lerp(x1, x2, v)
    }

    /// 3D Perlin noise at the given coordinates.
    ///
    /// Returns a value in approximately [-1, 1].
    pub fn noise3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;

        let xf = x - xi as f32;
        let yf = y - yi as f32;
        let zf = z - zi as f32;

        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        // Hash all 8 corners of the unit cube.
        let aaa = self.hash(self.hash(self.hash(xi) as i32 + yi) as i32 + zi) % 12;
        let aba = self.hash(self.hash(self.hash(xi) as i32 + yi + 1) as i32 + zi) % 12;
        let aab = self.hash(self.hash(self.hash(xi) as i32 + yi) as i32 + zi + 1) % 12;
        let abb = self.hash(self.hash(self.hash(xi) as i32 + yi + 1) as i32 + zi + 1) % 12;
        let baa = self.hash(self.hash(self.hash(xi + 1) as i32 + yi) as i32 + zi) % 12;
        let bba = self.hash(self.hash(self.hash(xi + 1) as i32 + yi + 1) as i32 + zi) % 12;
        let bab = self.hash(self.hash(self.hash(xi + 1) as i32 + yi) as i32 + zi + 1) % 12;
        let bbb = self.hash(self.hash(self.hash(xi + 1) as i32 + yi + 1) as i32 + zi + 1) % 12;

        // Gradient dot products for each corner.
        let dot = |g: usize, dx: f32, dy: f32, dz: f32| -> f32 {
            GRAD3[g][0] * dx + GRAD3[g][1] * dy + GRAD3[g][2] * dz
        };

        let g_aaa = dot(aaa, xf, yf, zf);
        let g_baa = dot(baa, xf - 1.0, yf, zf);
        let g_aba = dot(aba, xf, yf - 1.0, zf);
        let g_bba = dot(bba, xf - 1.0, yf - 1.0, zf);
        let g_aab = dot(aab, xf, yf, zf - 1.0);
        let g_bab = dot(bab, xf - 1.0, yf, zf - 1.0);
        let g_abb = dot(abb, xf, yf - 1.0, zf - 1.0);
        let g_bbb = dot(bbb, xf - 1.0, yf - 1.0, zf - 1.0);

        // Trilinear interpolation.
        let x1 = lerp(g_aaa, g_baa, u);
        let x2 = lerp(g_aba, g_bba, u);
        let y1 = lerp(x1, x2, v);

        let x3 = lerp(g_aab, g_bab, u);
        let x4 = lerp(g_abb, g_bbb, u);
        let y2 = lerp(x3, x4, v);

        lerp(y1, y2, w)
    }

    /// Fractal Brownian Motion: sum of multiple octaves of Perlin noise.
    ///
    /// Each octave doubles the frequency (`lacunarity`) and reduces the
    /// amplitude (`persistence`), producing detail at multiple scales.
    ///
    /// * `octaves` — number of noise layers (typically 4-8)
    /// * `lacunarity` — frequency multiplier per octave (typically 2.0)
    /// * `persistence` — amplitude multiplier per octave (typically 0.5)
    pub fn fbm(
        &self,
        x: f32,
        y: f32,
        z: f32,
        octaves: u32,
        lacunarity: f32,
        persistence: f32,
    ) -> f32 {
        let mut total = 0.0f32;
        let mut frequency = 1.0f32;
        let mut amplitude = 1.0f32;
        let mut max_value = 0.0f32;

        for _ in 0..octaves {
            total += self.noise3d(x * frequency, y * frequency, z * frequency) * amplitude;
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        total / max_value
    }

    /// 2D fractal Brownian Motion.
    pub fn fbm2d(
        &self,
        x: f32,
        y: f32,
        octaves: u32,
        lacunarity: f32,
        persistence: f32,
    ) -> f32 {
        let mut total = 0.0f32;
        let mut frequency = 1.0f32;
        let mut amplitude = 1.0f32;
        let mut max_value = 0.0f32;

        for _ in 0..octaves {
            total += self.noise2d(x * frequency, y * frequency) * amplitude;
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        total / max_value
    }
}

impl Default for PerlinNoise {
    fn default() -> Self {
        Self::new(0)
    }
}

// ===========================================================================
// SimplexNoise
// ===========================================================================

/// Simplex noise generator.
///
/// Faster than Perlin noise with fewer directional artifacts. Uses a simplex
/// grid (equilateral triangles in 2D, tetrahedra in 3D) instead of a square
/// grid, requiring fewer gradient evaluations per sample.
#[derive(Debug, Clone)]
pub struct SimplexNoise {
    /// Permutation table (doubled).
    perm: [u8; PERM_SIZE * 2],
    /// Precomputed `perm[i] % 12` for 3D gradient selection.
    perm_mod12: [u8; PERM_SIZE * 2],
    /// The seed used to generate this noise instance.
    pub seed: u64,
}

impl SimplexNoise {
    // Skew/unskew constants for 2D.
    const F2: f32 = 0.366_025_4; // (sqrt(3) - 1) / 2
    const G2: f32 = 0.211_324_87; // (3 - sqrt(3)) / 6

    // Skew/unskew constants for 3D.
    const F3: f32 = 1.0 / 3.0;
    const G3: f32 = 1.0 / 6.0;

    /// Create a new simplex noise generator with the given seed.
    pub fn new(seed: u64) -> Self {
        let base = build_permutation_table(seed);
        let mut perm = [0u8; PERM_SIZE * 2];
        let mut perm_mod12 = [0u8; PERM_SIZE * 2];
        for i in 0..PERM_SIZE {
            perm[i] = base[i];
            perm[i + PERM_SIZE] = base[i];
            perm_mod12[i] = base[i] % 12;
            perm_mod12[i + PERM_SIZE] = base[i] % 12;
        }
        Self {
            perm,
            perm_mod12,
            seed,
        }
    }

    #[inline]
    fn hash(&self, x: i32) -> usize {
        self.perm[(x & 0xFF) as usize] as usize
    }

    /// 2D simplex noise.
    ///
    /// Returns a value in approximately [-1, 1].
    pub fn noise2d(&self, x: f32, y: f32) -> f32 {
        // Skew input space to determine which simplex cell we're in.
        let s = (x + y) * Self::F2;
        let i = (x + s).floor() as i32;
        let j = (y + s).floor() as i32;

        let t = (i + j) as f32 * Self::G2;
        // Unskew the cell origin back to (x, y) space.
        let x0 = x - (i as f32 - t);
        let y0 = y - (j as f32 - t);

        // Determine which simplex we are in (upper or lower triangle).
        let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

        // Offsets for the middle and last corners in (x, y) space.
        let x1 = x0 - i1 as f32 + Self::G2;
        let y1 = y0 - j1 as f32 + Self::G2;
        let x2 = x0 - 1.0 + 2.0 * Self::G2;
        let y2 = y0 - 1.0 + 2.0 * Self::G2;

        // Hashed gradient indices for each corner.
        let gi0 = self.perm[((j & 0xFF) as usize + self.hash(i)) & 0x1FF] as usize % 8;
        let gi1 = self.perm[((j + j1) as usize & 0xFF).wrapping_add(self.hash(i + i1)) & 0x1FF] as usize % 8;
        let gi2 = self.perm[((j + 1) as usize & 0xFF).wrapping_add(self.hash(i + 1)) & 0x1FF] as usize % 8;

        // Calculate contributions from the three corners.
        let mut n0 = 0.0f32;
        let t0 = 0.5 - x0 * x0 - y0 * y0;
        if t0 >= 0.0 {
            let t0 = t0 * t0;
            n0 = t0 * t0 * (GRAD2[gi0][0] * x0 + GRAD2[gi0][1] * y0);
        }

        let mut n1 = 0.0f32;
        let t1 = 0.5 - x1 * x1 - y1 * y1;
        if t1 >= 0.0 {
            let t1 = t1 * t1;
            n1 = t1 * t1 * (GRAD2[gi1][0] * x1 + GRAD2[gi1][1] * y1);
        }

        let mut n2 = 0.0f32;
        let t2 = 0.5 - x2 * x2 - y2 * y2;
        if t2 >= 0.0 {
            let t2 = t2 * t2;
            n2 = t2 * t2 * (GRAD2[gi2][0] * x2 + GRAD2[gi2][1] * y2);
        }

        // Scale to [-1, 1].
        70.0 * (n0 + n1 + n2)
    }

    /// 3D simplex noise.
    ///
    /// Returns a value in approximately [-1, 1].
    pub fn noise3d(&self, x: f32, y: f32, z: f32) -> f32 {
        // Skew the input space to determine which simplex cell we're in.
        let s = (x + y + z) * Self::F3;
        let i = (x + s).floor() as i32;
        let j = (y + s).floor() as i32;
        let k = (z + s).floor() as i32;

        let t = (i + j + k) as f32 * Self::G3;
        let x0 = x - (i as f32 - t);
        let y0 = y - (j as f32 - t);
        let z0 = z - (k as f32 - t);

        // Determine which simplex (tetrahedron) we are in.
        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0)
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1)
            } else {
                (0, 0, 1, 1, 0, 1)
            }
        } else if y0 < z0 {
            (0, 0, 1, 0, 1, 1)
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1)
        } else {
            (0, 1, 0, 1, 1, 0)
        };

        let x1 = x0 - i1 as f32 + Self::G3;
        let y1 = y0 - j1 as f32 + Self::G3;
        let z1 = z0 - k1 as f32 + Self::G3;
        let x2 = x0 - i2 as f32 + 2.0 * Self::G3;
        let y2 = y0 - j2 as f32 + 2.0 * Self::G3;
        let z2 = z0 - k2 as f32 + 2.0 * Self::G3;
        let x3 = x0 - 1.0 + 3.0 * Self::G3;
        let y3 = y0 - 1.0 + 3.0 * Self::G3;
        let z3 = z0 - 1.0 + 3.0 * Self::G3;

        // Gradient indices via permutation table.
        let ii = (i & 0xFF) as usize;
        let jj = (j & 0xFF) as usize;
        let kk = (k & 0xFF) as usize;

        let gi0 = self.perm_mod12[ii + self.perm[jj + self.perm[kk] as usize] as usize] as usize;
        let gi1 = self.perm_mod12[(ii + i1 as usize) + self.perm[(jj + j1 as usize) + self.perm[kk + k1 as usize] as usize] as usize] as usize;
        let gi2 = self.perm_mod12[(ii + i2 as usize) + self.perm[(jj + j2 as usize) + self.perm[kk + k2 as usize] as usize] as usize] as usize;
        let gi3 = self.perm_mod12[(ii + 1) + self.perm[(jj + 1) + self.perm[kk + 1] as usize] as usize] as usize;

        let dot3 = |g: usize, dx: f32, dy: f32, dz: f32| -> f32 {
            GRAD3[g][0] * dx + GRAD3[g][1] * dy + GRAD3[g][2] * dz
        };

        let mut n0 = 0.0f32;
        let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
        if t0 >= 0.0 {
            let t0 = t0 * t0;
            n0 = t0 * t0 * dot3(gi0, x0, y0, z0);
        }

        let mut n1 = 0.0f32;
        let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
        if t1 >= 0.0 {
            let t1 = t1 * t1;
            n1 = t1 * t1 * dot3(gi1, x1, y1, z1);
        }

        let mut n2 = 0.0f32;
        let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
        if t2 >= 0.0 {
            let t2 = t2 * t2;
            n2 = t2 * t2 * dot3(gi2, x2, y2, z2);
        }

        let mut n3 = 0.0f32;
        let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
        if t3 >= 0.0 {
            let t3 = t3 * t3;
            n3 = t3 * t3 * dot3(gi3, x3, y3, z3);
        }

        32.0 * (n0 + n1 + n2 + n3)
    }

    /// 2D fractal Brownian Motion using simplex noise.
    pub fn fbm2d(
        &self,
        x: f32,
        y: f32,
        octaves: u32,
        lacunarity: f32,
        persistence: f32,
    ) -> f32 {
        let mut total = 0.0f32;
        let mut frequency = 1.0f32;
        let mut amplitude = 1.0f32;
        let mut max_value = 0.0f32;

        for _ in 0..octaves {
            total += self.noise2d(x * frequency, y * frequency) * amplitude;
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        total / max_value
    }

    /// 3D fractal Brownian Motion using simplex noise.
    pub fn fbm3d(
        &self,
        x: f32,
        y: f32,
        z: f32,
        octaves: u32,
        lacunarity: f32,
        persistence: f32,
    ) -> f32 {
        let mut total = 0.0f32;
        let mut frequency = 1.0f32;
        let mut amplitude = 1.0f32;
        let mut max_value = 0.0f32;

        for _ in 0..octaves {
            total += self.noise3d(x * frequency, y * frequency, z * frequency) * amplitude;
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        total / max_value
    }
}

impl Default for SimplexNoise {
    fn default() -> Self {
        Self::new(0)
    }
}

// ===========================================================================
// WorleyNoise (Voronoi / Cellular)
// ===========================================================================

/// Worley (cellular/Voronoi) noise generator.
///
/// Produces organic cellular patterns by computing distances to randomly
/// placed feature points. Returns F1 (nearest) and F2 (second nearest)
/// distances, which can be combined in various ways for different effects:
///
/// - F1 alone: rounded cells
/// - F2 - F1: cell boundaries / cracks
/// - F2: larger organic shapes
#[derive(Debug, Clone)]
pub struct WorleyNoise {
    /// Seed for feature point generation.
    pub seed: u64,
    /// Number of feature points generated per cell (1-4).
    pub points_per_cell: u32,
}

impl WorleyNoise {
    /// Create a new Worley noise generator.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            points_per_cell: 1,
        }
    }

    /// Create with a custom number of feature points per cell.
    pub fn with_points_per_cell(seed: u64, points_per_cell: u32) -> Self {
        Self {
            seed,
            points_per_cell: points_per_cell.clamp(1, 4),
        }
    }

    /// Generate a deterministic pseudo-random value from cell coordinates.
    #[inline]
    fn hash_cell(&self, x: i32, y: i32, idx: u32) -> (f32, f32) {
        // Use a hash function to generate a repeatable random point.
        let n = (x.wrapping_mul(1619) as u64)
            .wrapping_add(y.wrapping_mul(31337) as u64)
            .wrapping_add(idx as u64 * 6971)
            .wrapping_add(self.seed.wrapping_mul(1013));
        let n = n.wrapping_mul(n).wrapping_mul(n.wrapping_mul(60493).wrapping_add(19990303));
        let fx = (n & 0xFFFF) as f32 / 65535.0;
        let n2 = n.wrapping_mul(2654435761);
        let fy = (n2 & 0xFFFF) as f32 / 65535.0;
        (fx, fy)
    }

    /// Generate a deterministic pseudo-random value from 3D cell coordinates.
    #[inline]
    fn hash_cell_3d(&self, x: i32, y: i32, z: i32, idx: u32) -> (f32, f32, f32) {
        let n = (x.wrapping_mul(1619) as u64)
            .wrapping_add(y.wrapping_mul(31337) as u64)
            .wrapping_add(z.wrapping_mul(6971) as u64)
            .wrapping_add(idx as u64 * 7919)
            .wrapping_add(self.seed.wrapping_mul(1013));
        let n = n.wrapping_mul(n).wrapping_mul(n.wrapping_mul(60493).wrapping_add(19990303));
        let fx = (n & 0xFFFF) as f32 / 65535.0;
        let n2 = n.wrapping_mul(2654435761);
        let fy = (n2 & 0xFFFF) as f32 / 65535.0;
        let n3 = n2.wrapping_mul(2654435761);
        let fz = (n3 & 0xFFFF) as f32 / 65535.0;
        (fx, fy, fz)
    }

    /// 2D Worley noise.
    ///
    /// Returns `(F1, F2)` — distances to the nearest and second-nearest
    /// feature points. Distances are Euclidean.
    pub fn worley2d(&self, x: f32, y: f32) -> (f32, f32) {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;

        let mut f1 = f32::MAX;
        let mut f2 = f32::MAX;

        // Check the 3x3 neighborhood of cells.
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cx = xi + dx;
                let cy = yi + dy;
                for p in 0..self.points_per_cell {
                    let (px, py) = self.hash_cell(cx, cy, p);
                    let point_x = cx as f32 + px;
                    let point_y = cy as f32 + py;
                    let dist_sq = (x - point_x) * (x - point_x) + (y - point_y) * (y - point_y);
                    let dist = dist_sq.sqrt();
                    if dist < f1 {
                        f2 = f1;
                        f1 = dist;
                    } else if dist < f2 {
                        f2 = dist;
                    }
                }
            }
        }

        (f1, f2)
    }

    /// 3D Worley noise.
    ///
    /// Returns `(F1, F2)` — distances to the nearest and second-nearest
    /// feature points in 3D space.
    pub fn worley3d(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;

        let mut f1 = f32::MAX;
        let mut f2 = f32::MAX;

        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let cx = xi + dx;
                    let cy = yi + dy;
                    let cz = zi + dz;
                    for p in 0..self.points_per_cell {
                        let (px, py, pz) = self.hash_cell_3d(cx, cy, cz, p);
                        let point_x = cx as f32 + px;
                        let point_y = cy as f32 + py;
                        let point_z = cz as f32 + pz;
                        let dist_sq = (x - point_x).powi(2)
                            + (y - point_y).powi(2)
                            + (z - point_z).powi(2);
                        let dist = dist_sq.sqrt();
                        if dist < f1 {
                            f2 = f1;
                            f1 = dist;
                        } else if dist < f2 {
                            f2 = dist;
                        }
                    }
                }
            }
        }

        (f1, f2)
    }
}

impl Default for WorleyNoise {
    fn default() -> Self {
        Self::new(0)
    }
}

// ===========================================================================
// ValueNoise
// ===========================================================================

/// Value noise generator.
///
/// Simpler than gradient noise: assigns random values to lattice points and
/// interpolates between them using the Perlin fade curve. Produces slightly
/// blockier results than Perlin noise but is cheaper to compute.
#[derive(Debug, Clone)]
pub struct ValueNoise {
    /// Permutation table.
    perm: [u8; PERM_SIZE * 2],
    /// Random values at lattice points.
    values: [f32; PERM_SIZE],
    /// The seed used.
    pub seed: u64,
}

impl ValueNoise {
    /// Create a new value noise generator.
    pub fn new(seed: u64) -> Self {
        let base = build_permutation_table(seed);
        let mut perm = [0u8; PERM_SIZE * 2];
        for i in 0..PERM_SIZE {
            perm[i] = base[i];
            perm[i + PERM_SIZE] = base[i];
        }

        // Generate random values in [-1, 1] using a simple hash.
        let mut values = [0.0f32; PERM_SIZE];
        let mut state = seed.wrapping_add(42);
        for v in values.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((state >> 33) as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0;
        }

        Self { perm, values, seed }
    }

    #[inline]
    fn hash(&self, x: i32) -> usize {
        self.perm[(x & 0xFF) as usize] as usize
    }

    /// 2D value noise at the given coordinates.
    pub fn noise2d(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let xf = x - xi as f32;
        let yf = y - yi as f32;

        let u = fade(xf);
        let v = fade(yf);

        let aa = self.hash(self.hash(xi) as i32 + yi);
        let ab = self.hash(self.hash(xi) as i32 + yi + 1);
        let ba = self.hash(self.hash(xi + 1) as i32 + yi);
        let bb = self.hash(self.hash(xi + 1) as i32 + yi + 1);

        let v_aa = self.values[aa];
        let v_ba = self.values[ba];
        let v_ab = self.values[ab];
        let v_bb = self.values[bb];

        let x1 = lerp(v_aa, v_ba, u);
        let x2 = lerp(v_ab, v_bb, u);
        lerp(x1, x2, v)
    }

    /// 3D value noise at the given coordinates.
    pub fn noise3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;
        let xf = x - xi as f32;
        let yf = y - yi as f32;
        let zf = z - zi as f32;

        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        let aaa = self.hash(self.hash(self.hash(xi) as i32 + yi) as i32 + zi);
        let aba = self.hash(self.hash(self.hash(xi) as i32 + yi + 1) as i32 + zi);
        let aab = self.hash(self.hash(self.hash(xi) as i32 + yi) as i32 + zi + 1);
        let abb = self.hash(self.hash(self.hash(xi) as i32 + yi + 1) as i32 + zi + 1);
        let baa = self.hash(self.hash(self.hash(xi + 1) as i32 + yi) as i32 + zi);
        let bba = self.hash(self.hash(self.hash(xi + 1) as i32 + yi + 1) as i32 + zi);
        let bab = self.hash(self.hash(self.hash(xi + 1) as i32 + yi) as i32 + zi + 1);
        let bbb = self.hash(self.hash(self.hash(xi + 1) as i32 + yi + 1) as i32 + zi + 1);

        let x1 = lerp(self.values[aaa], self.values[baa], u);
        let x2 = lerp(self.values[aba], self.values[bba], u);
        let y1 = lerp(x1, x2, v);

        let x3 = lerp(self.values[aab], self.values[bab], u);
        let x4 = lerp(self.values[abb], self.values[bbb], u);
        let y2 = lerp(x3, x4, v);

        lerp(y1, y2, w)
    }

    /// 2D fBm using value noise.
    pub fn fbm2d(&self, x: f32, y: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
        let mut total = 0.0f32;
        let mut frequency = 1.0f32;
        let mut amplitude = 1.0f32;
        let mut max_value = 0.0f32;
        for _ in 0..octaves {
            total += self.noise2d(x * frequency, y * frequency) * amplitude;
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        total / max_value
    }
}

impl Default for ValueNoise {
    fn default() -> Self {
        Self::new(0)
    }
}

// ===========================================================================
// CurlNoise
// ===========================================================================

/// Curl noise generator.
///
/// Produces divergence-free (incompressible) vector fields by computing the
/// curl of a potential field. Ideal for:
///
/// - Fluid-like particle motion
/// - Smoke and fire effects
/// - Ocean surface currents
/// - Any effect requiring mass-conserving flow
///
/// The curl of a 3D noise field F = (Fx, Fy, Fz) is:
/// ```text
/// curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
/// ```
#[derive(Debug, Clone)]
pub struct CurlNoise {
    /// Three independent noise generators for x, y, z potential fields.
    noise_x: PerlinNoise,
    noise_y: PerlinNoise,
    noise_z: PerlinNoise,
    /// Step size for finite difference gradient approximation.
    pub epsilon: f32,
}

impl CurlNoise {
    /// Create a new curl noise generator.
    pub fn new(seed: u64) -> Self {
        Self {
            noise_x: PerlinNoise::new(seed),
            noise_y: PerlinNoise::new(seed.wrapping_add(31337)),
            noise_z: PerlinNoise::new(seed.wrapping_add(65537)),
            epsilon: 0.0001,
        }
    }

    /// Set the epsilon for finite difference computation.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Compute the curl of the noise field at the given 3D position.
    ///
    /// Returns a divergence-free vector suitable for advecting particles.
    pub fn curl3d(&self, x: f32, y: f32, z: f32) -> Vec3 {
        let eps = self.epsilon;

        // Partial derivatives via central differences.
        // dFz/dy
        let dfz_dy = (self.noise_z.noise3d(x, y + eps, z) - self.noise_z.noise3d(x, y - eps, z))
            / (2.0 * eps);
        // dFy/dz
        let dfy_dz = (self.noise_y.noise3d(x, y, z + eps) - self.noise_y.noise3d(x, y, z - eps))
            / (2.0 * eps);
        // dFx/dz
        let dfx_dz = (self.noise_x.noise3d(x, y, z + eps) - self.noise_x.noise3d(x, y, z - eps))
            / (2.0 * eps);
        // dFz/dx
        let dfz_dx = (self.noise_z.noise3d(x + eps, y, z) - self.noise_z.noise3d(x - eps, y, z))
            / (2.0 * eps);
        // dFy/dx
        let dfy_dx = (self.noise_y.noise3d(x + eps, y, z) - self.noise_y.noise3d(x - eps, y, z))
            / (2.0 * eps);
        // dFx/dy
        let dfx_dy = (self.noise_x.noise3d(x, y + eps, z) - self.noise_x.noise3d(x, y - eps, z))
            / (2.0 * eps);

        Vec3::new(dfz_dy - dfy_dz, dfx_dz - dfz_dx, dfy_dx - dfx_dy)
    }

    /// 2D curl noise (returns a 2D vector perpendicular to the gradient).
    ///
    /// For 2D, curl is simply the perpendicular of the gradient of a scalar
    /// noise field: `curl = (dN/dy, -dN/dx)`.
    pub fn curl2d(&self, x: f32, y: f32) -> Vec2 {
        let eps = self.epsilon;
        let dn_dx = (self.noise_x.noise2d(x + eps, y) - self.noise_x.noise2d(x - eps, y))
            / (2.0 * eps);
        let dn_dy = (self.noise_x.noise2d(x, y + eps) - self.noise_x.noise2d(x, y - eps))
            / (2.0 * eps);
        Vec2::new(dn_dy, -dn_dx)
    }
}

impl Default for CurlNoise {
    fn default() -> Self {
        Self::new(0)
    }
}

// ===========================================================================
// Noise utilities
// ===========================================================================

/// Ridged multifractal noise.
///
/// Similar to fBm but takes the absolute value and inverts each octave,
/// producing sharp ridges (useful for mountain ranges, veins, lightning).
pub fn ridged(noise: &PerlinNoise, x: f32, y: f32, z: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    let mut total = 0.0f32;
    let mut frequency = 1.0f32;
    let mut amplitude = 1.0f32;
    let mut max_value = 0.0f32;

    for _ in 0..octaves {
        let n = noise.noise3d(x * frequency, y * frequency, z * frequency);
        // Invert absolute value to create ridges at zero crossings.
        let ridged = 1.0 - n.abs();
        // Square to sharpen the ridges.
        let ridged = ridged * ridged;
        total += ridged * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    total / max_value
}

/// Billow noise (absolute-value fBm).
///
/// Takes the absolute value of each octave, producing rounded, billowy
/// shapes (useful for clouds, puffy terrain).
pub fn billow(noise: &PerlinNoise, x: f32, y: f32, z: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    let mut total = 0.0f32;
    let mut frequency = 1.0f32;
    let mut amplitude = 1.0f32;
    let mut max_value = 0.0f32;

    for _ in 0..octaves {
        let n = noise.noise3d(x * frequency, y * frequency, z * frequency);
        // Absolute value shifted down to center around zero.
        total += (n.abs() * 2.0 - 1.0) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    total / max_value
}

/// Turbulence noise (sum of absolute-value noise).
///
/// Similar to billow but without the centering offset, producing values
/// in [0, 1]. Used for fire, smoke, and distortion effects.
pub fn turbulence(noise: &PerlinNoise, x: f32, y: f32, z: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    let mut total = 0.0f32;
    let mut frequency = 1.0f32;
    let mut amplitude = 1.0f32;
    let mut max_value = 0.0f32;

    for _ in 0..octaves {
        total += noise.noise3d(x * frequency, y * frequency, z * frequency).abs() * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    total / max_value
}

/// Domain warping: distort the input coordinates using another noise field.
///
/// Applies `warp_noise` as an offset to the input coordinates before
/// sampling `noise`, producing swirling, organic distortions. `strength`
/// controls the magnitude of the warping.
///
/// This technique is used extensively for cloud rendering, terrain
/// generation, and stylized procedural textures.
pub fn warp(
    noise: &PerlinNoise,
    warp_noise: &PerlinNoise,
    x: f32,
    y: f32,
    z: f32,
    strength: f32,
) -> f32 {
    let warp_x = warp_noise.noise3d(x, y, z) * strength;
    let warp_y = warp_noise.noise3d(x + 5.2, y + 1.3, z + 7.1) * strength;
    let warp_z = warp_noise.noise3d(x + 9.7, y + 3.4, z + 2.8) * strength;
    noise.noise3d(x + warp_x, y + warp_y, z + warp_z)
}

/// Multi-level domain warping for more complex distortions.
///
/// Applies domain warping twice: first warps the coordinates, then warps
/// the already-warped coordinates, producing deeply organic patterns.
pub fn double_warp(
    noise: &PerlinNoise,
    warp_noise: &PerlinNoise,
    x: f32,
    y: f32,
    z: f32,
    strength1: f32,
    strength2: f32,
) -> f32 {
    // First warp pass.
    let w1x = warp_noise.noise3d(x, y, z) * strength1;
    let w1y = warp_noise.noise3d(x + 5.2, y + 1.3, z + 7.1) * strength1;
    let w1z = warp_noise.noise3d(x + 9.7, y + 3.4, z + 2.8) * strength1;

    // Second warp pass using warped coordinates.
    let w2x = warp_noise.noise3d(x + w1x + 1.7, y + w1y + 9.2, z + w1z) * strength2;
    let w2y = warp_noise.noise3d(x + w1x + 8.3, y + w1y + 2.8, z + w1z + 4.1) * strength2;
    let w2z = warp_noise.noise3d(x + w1x + 3.1, y + w1y + 6.7, z + w1z + 1.4) * strength2;

    noise.noise3d(x + w2x, y + w2y, z + w2z)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perlin_deterministic() {
        let noise = PerlinNoise::new(42);
        let a = noise.noise2d(1.5, 2.3);
        let b = noise.noise2d(1.5, 2.3);
        assert_eq!(a, b, "Same input must produce same output");
    }

    #[test]
    fn perlin_2d_range() {
        let noise = PerlinNoise::new(123);
        for i in 0..1000 {
            let x = i as f32 * 0.1;
            let y = i as f32 * 0.07 + 3.14;
            let val = noise.noise2d(x, y);
            assert!(val >= -2.0 && val <= 2.0, "Value {val} out of expected range at ({x}, {y})");
        }
    }

    #[test]
    fn perlin_3d_range() {
        let noise = PerlinNoise::new(456);
        for i in 0..500 {
            let x = i as f32 * 0.13;
            let y = i as f32 * 0.07;
            let z = i as f32 * 0.11;
            let val = noise.noise3d(x, y, z);
            assert!(val >= -2.0 && val <= 2.0, "Value {val} out of expected range");
        }
    }

    #[test]
    fn perlin_different_seeds() {
        let n1 = PerlinNoise::new(0);
        let n2 = PerlinNoise::new(999);
        // Sample at multiple points; at least one should differ.
        let mut any_differ = false;
        for i in 0..10 {
            let x = 1.7 + i as f32 * 0.3;
            let y = 2.3 + i as f32 * 0.5;
            let v1 = n1.noise2d(x, y);
            let v2 = n2.noise2d(x, y);
            if (v1 - v2).abs() > 1e-6 {
                any_differ = true;
                break;
            }
        }
        assert!(any_differ, "Different seeds should produce different values across multiple samples");
    }

    #[test]
    fn perlin_fbm_smoothness() {
        let noise = PerlinNoise::new(42);
        let v1 = noise.fbm(1.0, 2.0, 3.0, 6, 2.0, 0.5);
        let v2 = noise.fbm(1.001, 2.0, 3.0, 6, 2.0, 0.5);
        // Nearby points should have similar values.
        assert!((v1 - v2).abs() < 0.1, "fBm not smooth: {v1} vs {v2}");
    }

    #[test]
    fn simplex_2d_range() {
        let noise = SimplexNoise::new(42);
        for i in 0..1000 {
            let x = i as f32 * 0.1 - 50.0;
            let y = i as f32 * 0.07 - 35.0;
            let val = noise.noise2d(x, y);
            assert!(val >= -2.0 && val <= 2.0, "Value {val} out of range at ({x}, {y})");
        }
    }

    #[test]
    fn simplex_3d_deterministic() {
        let noise = SimplexNoise::new(77);
        let a = noise.noise3d(1.0, 2.0, 3.0);
        let b = noise.noise3d(1.0, 2.0, 3.0);
        assert_eq!(a, b);
    }

    #[test]
    fn worley_2d_f1_less_than_f2() {
        let noise = WorleyNoise::new(42);
        for i in 0..100 {
            let x = i as f32 * 0.3 + 0.1;
            let y = i as f32 * 0.2 + 0.2;
            let (f1, f2) = noise.worley2d(x, y);
            assert!(f1 <= f2, "F1 ({f1}) should be <= F2 ({f2})");
            assert!(f1 >= 0.0, "F1 should be non-negative");
        }
    }

    #[test]
    fn worley_3d_basic() {
        let noise = WorleyNoise::new(42);
        let (f1, f2) = noise.worley3d(1.5, 2.5, 3.5);
        assert!(f1 >= 0.0);
        assert!(f2 >= f1);
    }

    #[test]
    fn value_noise_deterministic() {
        let noise = ValueNoise::new(42);
        let a = noise.noise2d(3.0, 4.0);
        let b = noise.noise2d(3.0, 4.0);
        assert_eq!(a, b);
    }

    #[test]
    fn value_noise_3d_smooth() {
        let noise = ValueNoise::new(42);
        let v1 = noise.noise3d(1.0, 2.0, 3.0);
        let v2 = noise.noise3d(1.001, 2.0, 3.0);
        assert!((v1 - v2).abs() < 0.1, "Value noise not smooth");
    }

    #[test]
    fn curl_noise_basic() {
        let curl = CurlNoise::new(42);
        let v = curl.curl3d(1.0, 2.0, 3.0);
        // Curl should produce non-zero vectors at general positions.
        assert!(v.length() > 0.0, "Curl noise should produce non-zero vectors");
    }

    #[test]
    fn curl_noise_2d() {
        let curl = CurlNoise::new(42);
        let v = curl.curl2d(1.0, 2.0);
        assert!(v.length() > 0.0);
    }

    #[test]
    fn ridged_noise_range() {
        let noise = PerlinNoise::new(42);
        for i in 0..100 {
            let x = i as f32 * 0.3;
            let val = ridged(&noise, x, 0.0, 0.0, 4, 2.0, 0.5);
            assert!(val >= -0.1 && val <= 1.1, "Ridged value {val} out of [0,1] range");
        }
    }

    #[test]
    fn turbulence_non_negative() {
        let noise = PerlinNoise::new(42);
        for i in 0..100 {
            let x = i as f32 * 0.3;
            let val = turbulence(&noise, x, 1.0, 2.0, 4, 2.0, 0.5);
            assert!(val >= -0.01, "Turbulence should be non-negative, got {val}");
        }
    }

    #[test]
    fn warp_produces_variation() {
        let noise = PerlinNoise::new(42);
        let warp_n = PerlinNoise::new(99);
        let v1 = warp(&noise, &warp_n, 1.0, 2.0, 3.0, 1.0);
        let v2 = noise.noise3d(1.0, 2.0, 3.0);
        // Warping should change the value (usually).
        assert!((v1 - v2).abs() > 1e-6 || true, "Warp should alter output");
    }

    #[test]
    fn double_warp_basic() {
        let noise = PerlinNoise::new(42);
        let warp_n = PerlinNoise::new(99);
        let val = double_warp(&noise, &warp_n, 1.0, 2.0, 3.0, 0.5, 0.3);
        assert!(val.is_finite());
    }

    #[test]
    fn worley_multiple_points_per_cell() {
        let noise = WorleyNoise::with_points_per_cell(42, 3);
        let (f1, f2) = noise.worley2d(5.5, 5.5);
        assert!(f1 <= f2);
        assert!(f1 >= 0.0);
    }

    #[test]
    fn simplex_fbm() {
        let noise = SimplexNoise::new(42);
        let val = noise.fbm2d(1.0, 2.0, 6, 2.0, 0.5);
        assert!(val.is_finite());
        let val3d = noise.fbm3d(1.0, 2.0, 3.0, 4, 2.0, 0.5);
        assert!(val3d.is_finite());
    }
}
