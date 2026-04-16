//! Random number generation for the Genovo engine.
//!
//! Provides a fast, deterministic PRNG (xoshiro256**) and quasi-random
//! sampling utilities. The generator is not cryptographically secure but
//! is suitable for game logic, procedural generation, and simulation.
//!
//! # Determinism
//!
//! All generators are fully deterministic given the same seed, which is
//! critical for:
//! - Reproducible procedural worlds
//! - Network synchronization (lockstep)
//! - Replay systems
//! - Automated testing

use glam::{Vec2, Vec3};

// ===========================================================================
// Rng — xoshiro256** PRNG
// ===========================================================================

/// A fast pseudo-random number generator using the xoshiro256** algorithm.
///
/// xoshiro256** has a period of 2^256 - 1, excellent statistical properties,
/// and is one of the fastest high-quality PRNGs available. It passes all
/// tests in the BigCrush and PractRand test suites.
///
/// # Example
/// ```ignore
/// let mut rng = Rng::new(42);
/// let roll = rng.range_i32(1, 7); // 1d6
/// let chance = rng.next_f32(); // [0, 1)
/// ```
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Create a new RNG with the given seed.
    ///
    /// The seed is expanded into the full 256-bit state using SplitMix64
    /// to ensure good initial state even from simple seeds like 0 or 1.
    pub fn new(seed: u64) -> Self {
        // Initialize with SplitMix64 to expand the seed.
        let mut sm = seed;
        let mut state = [0u64; 4];
        for s in state.iter_mut() {
            sm = sm.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            *s = z;
        }
        // Ensure state is not all zeros (xoshiro requirement).
        if state.iter().all(|&s| s == 0) {
            state[0] = 1;
        }
        Self { state }
    }

    /// Generate the next 64-bit random value.
    ///
    /// This is the raw output of the xoshiro256** algorithm.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);

        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Generate a random 32-bit unsigned integer.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Generate a random f32 in [0, 1).
    ///
    /// Uses the upper 23 bits of a 64-bit value for maximum precision
    /// in the mantissa of the IEEE 754 f32 format.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Use upper bits (better quality in xoshiro).
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Generate a random f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate a random f32 in [min, max).
    #[inline]
    pub fn range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }

    /// Generate a random i32 in [min, max) (exclusive upper bound).
    pub fn range_i32(&mut self, min: i32, max: i32) -> i32 {
        if min >= max {
            return min;
        }
        let range = (max - min) as u32;
        // Debiased modulo using rejection sampling.
        let threshold = range.wrapping_neg() % range;
        loop {
            let val = self.next_u32();
            if val >= threshold {
                return min + (val % range) as i32;
            }
        }
    }

    /// Generate a random boolean with the given probability of being true.
    ///
    /// `probability` is in [0, 1] where 0 = always false, 1 = always true.
    #[inline]
    pub fn bool(&mut self, probability: f32) -> bool {
        self.next_f32() < probability
    }

    /// Generate a random point on the unit circle (radius = 1).
    ///
    /// Uses rejection sampling for uniform distribution.
    pub fn unit_circle(&mut self) -> Vec2 {
        let angle = self.next_f32() * std::f32::consts::TAU;
        Vec2::new(angle.cos(), angle.sin())
    }

    /// Generate a random point inside a circle of the given radius.
    pub fn in_circle(&mut self, radius: f32) -> Vec2 {
        // Use sqrt for uniform area distribution.
        let r = radius * self.next_f32().sqrt();
        let angle = self.next_f32() * std::f32::consts::TAU;
        Vec2::new(r * angle.cos(), r * angle.sin())
    }

    /// Generate a random point on the unit sphere (radius = 1).
    ///
    /// Uses the Marsaglia method for uniform distribution on the sphere.
    pub fn unit_sphere(&mut self) -> Vec3 {
        loop {
            let x = self.range(-1.0, 1.0);
            let y = self.range(-1.0, 1.0);
            let s = x * x + y * y;
            if s < 1.0 && s > f32::EPSILON {
                let factor = 2.0 * (1.0 - s).sqrt();
                return Vec3::new(x * factor, y * factor, 1.0 - 2.0 * s);
            }
        }
    }

    /// Generate a random point inside a sphere of the given radius.
    ///
    /// Uses rejection sampling for uniform volume distribution.
    pub fn in_sphere(&mut self, radius: f32) -> Vec3 {
        loop {
            let x = self.range(-radius, radius);
            let y = self.range(-radius, radius);
            let z = self.range(-radius, radius);
            if x * x + y * y + z * z <= radius * radius {
                return Vec3::new(x, y, z);
            }
        }
    }

    /// Generate a random 2D direction vector (unit length).
    pub fn direction_2d(&mut self) -> Vec2 {
        self.unit_circle()
    }

    /// Generate a random 3D direction vector (unit length).
    pub fn direction_3d(&mut self) -> Vec3 {
        self.unit_sphere()
    }

    /// Generate a random point in a hemisphere around the given normal.
    ///
    /// Useful for ambient occlusion sampling and diffuse lighting.
    pub fn hemisphere(&mut self, normal: Vec3) -> Vec3 {
        let dir = self.unit_sphere();
        if dir.dot(normal) < 0.0 {
            -dir
        } else {
            dir
        }
    }

    /// Cosine-weighted hemisphere sampling.
    ///
    /// Generates samples with a distribution proportional to cos(theta),
    /// which is optimal for diffuse lighting Monte Carlo integration.
    pub fn cosine_hemisphere(&mut self, normal: Vec3) -> Vec3 {
        let u1 = self.next_f32();
        let u2 = self.next_f32();

        let r = u1.sqrt();
        let theta = std::f32::consts::TAU * u2;

        let x = r * theta.cos();
        let y = r * theta.sin();
        let z = (1.0 - u1).sqrt();

        // Build tangent space from normal.
        let up = if normal.y.abs() < 0.999 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let tangent = normal.cross(up).normalize();
        let bitangent = normal.cross(tangent);

        tangent * x + bitangent * y + normal * z
    }

    /// Shuffle a slice in-place using Fisher-Yates algorithm.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = self.range_i32(0, (i + 1) as i32) as usize;
            slice.swap(i, j);
        }
    }

    /// Pick a random element from a slice.
    ///
    /// Returns `None` if the slice is empty.
    pub fn pick<'a, T>(&mut self, slice: &'a [T]) -> Option<&'a T> {
        if slice.is_empty() {
            return None;
        }
        let idx = self.range_i32(0, slice.len() as i32) as usize;
        Some(&slice[idx])
    }

    /// Pick a random element with weighted probabilities.
    ///
    /// `weights` provides relative probabilities for each element. Returns
    /// the index of the selected element. Weights do not need to sum to 1.
    ///
    /// Returns 0 if weights is empty.
    pub fn weighted_pick(&mut self, weights: &[f32]) -> usize {
        if weights.is_empty() {
            return 0;
        }
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return self.range_i32(0, weights.len() as i32) as usize;
        }

        let mut target = self.next_f32() * total;
        for (i, &w) in weights.iter().enumerate() {
            target -= w;
            if target <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }

    /// Generate a normally distributed random value (Box-Muller transform).
    ///
    /// Returns a value with mean 0 and standard deviation 1.
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::EPSILON);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Generate a normally distributed value with the given mean and std dev.
    pub fn gaussian_range(&mut self, mean: f32, std_dev: f32) -> f32 {
        mean + self.gaussian() * std_dev
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(0)
    }
}

// ===========================================================================
// Halton sequence
// ===========================================================================

/// A Halton sequence generator for quasi-random sampling.
///
/// Halton sequences produce low-discrepancy point sets that cover the
/// sampling domain more uniformly than pseudo-random numbers. They are
/// commonly used for:
///
/// - Anti-aliasing (temporal AA jitter)
/// - Monte Carlo integration
/// - Blue-noise-like sample distribution
/// - Quasi-random texture sampling
#[derive(Debug, Clone)]
pub struct Halton {
    /// Current index in the sequence.
    index: u32,
    /// Base for the first dimension.
    base1: u32,
    /// Base for the second dimension.
    base2: u32,
}

impl Halton {
    /// Create a new Halton sequence generator.
    ///
    /// Uses base 2 and base 3 by default (the most common choice).
    pub fn new() -> Self {
        Self {
            index: 0,
            base1: 2,
            base2: 3,
        }
    }

    /// Create with custom bases.
    ///
    /// Bases should be coprime (typically consecutive primes: 2,3 or 2,5).
    pub fn with_bases(base1: u32, base2: u32) -> Self {
        Self {
            index: 0,
            base1: base1.max(2),
            base2: base2.max(2),
        }
    }

    /// Compute the Halton value for the given index and base.
    ///
    /// This is the radical inverse function: reverses the digits of `index`
    /// in the given base to produce a value in [0, 1).
    fn radical_inverse(mut index: u32, base: u32) -> f32 {
        let mut result = 0.0f32;
        let mut f = 1.0f32 / base as f32;
        while index > 0 {
            result += f * (index % base) as f32;
            index /= base;
            f /= base as f32;
        }
        result
    }

    /// Generate the next 2D sample.
    pub fn next(&mut self) -> Vec2 {
        self.index += 1;
        Vec2::new(
            Self::radical_inverse(self.index, self.base1),
            Self::radical_inverse(self.index, self.base2),
        )
    }

    /// Generate a specific sample by index.
    pub fn sample(&self, index: u32) -> Vec2 {
        Vec2::new(
            Self::radical_inverse(index + 1, self.base1),
            Self::radical_inverse(index + 1, self.base2),
        )
    }

    /// Generate the next 1D sample (first dimension only).
    pub fn next_1d(&mut self) -> f32 {
        self.index += 1;
        Self::radical_inverse(self.index, self.base1)
    }

    /// Reset the sequence to the beginning.
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Get the current index.
    pub fn current_index(&self) -> u32 {
        self.index
    }
}

impl Default for Halton {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// PoissonDisk sampling
// ===========================================================================

/// 2D Poisson disk sampling using Bridson's algorithm.
///
/// Generates a set of points where no two points are closer than a minimum
/// distance `r`. This produces a natural, organic-looking distribution
/// similar to blue noise.
///
/// Common uses:
/// - Object placement (trees, buildings, enemies)
/// - Stippling/dithering patterns
/// - Sample point generation for AO/GI
#[derive(Debug, Clone)]
pub struct PoissonDisk {
    /// Minimum distance between points.
    pub min_distance: f32,
    /// Width of the sampling region.
    pub width: f32,
    /// Height of the sampling region.
    pub height: f32,
    /// Number of candidates to test before rejecting a point.
    pub max_attempts: u32,
}

impl PoissonDisk {
    /// Create a new Poisson disk sampler.
    ///
    /// * `min_distance` — minimum distance between any two points
    /// * `width` — width of the sampling region
    /// * `height` — height of the sampling region
    pub fn new(min_distance: f32, width: f32, height: f32) -> Self {
        Self {
            min_distance,
            width,
            height,
            max_attempts: 30,
        }
    }

    /// Set the maximum number of candidate attempts per active point.
    pub fn with_max_attempts(mut self, attempts: u32) -> Self {
        self.max_attempts = attempts.max(1);
        self
    }

    /// Generate a set of Poisson-disk-distributed points.
    ///
    /// Uses Bridson's algorithm, which runs in O(n) time where n is the
    /// number of output points.
    pub fn generate(&self, rng: &mut Rng) -> Vec<Vec2> {
        let r = self.min_distance;
        let cell_size = r / std::f32::consts::SQRT_2;

        let grid_width = (self.width / cell_size).ceil() as usize + 1;
        let grid_height = (self.height / cell_size).ceil() as usize + 1;

        // Background grid for spatial acceleration.
        // Each cell stores the index of the point in that cell, or -1.
        let mut grid = vec![-1i32; grid_width * grid_height];

        let mut points = Vec::new();
        let mut active_list: Vec<usize> = Vec::new();

        // Start with a random initial point.
        let initial = Vec2::new(
            rng.range(0.0, self.width),
            rng.range(0.0, self.height),
        );
        points.push(initial);
        active_list.push(0);

        let gx = (initial.x / cell_size) as usize;
        let gy = (initial.y / cell_size) as usize;
        if gx < grid_width && gy < grid_height {
            grid[gy * grid_width + gx] = 0;
        }

        while !active_list.is_empty() {
            // Pick a random active point.
            let active_idx = rng.range_i32(0, active_list.len() as i32) as usize;
            let center = points[active_list[active_idx]];

            let mut found = false;

            for _ in 0..self.max_attempts {
                // Generate a candidate point in the annulus [r, 2r] around center.
                let angle = rng.next_f32() * std::f32::consts::TAU;
                let dist = r + rng.next_f32() * r;
                let candidate = Vec2::new(
                    center.x + angle.cos() * dist,
                    center.y + angle.sin() * dist,
                );

                // Check bounds.
                if candidate.x < 0.0
                    || candidate.x >= self.width
                    || candidate.y < 0.0
                    || candidate.y >= self.height
                {
                    continue;
                }

                let cx = (candidate.x / cell_size) as usize;
                let cy = (candidate.y / cell_size) as usize;

                // Check the 5x5 neighborhood in the grid.
                let mut too_close = false;
                let start_x = cx.saturating_sub(2);
                let start_y = cy.saturating_sub(2);
                let end_x = (cx + 3).min(grid_width);
                let end_y = (cy + 3).min(grid_height);

                'check: for ny in start_y..end_y {
                    for nx in start_x..end_x {
                        let cell_idx = ny * grid_width + nx;
                        if grid[cell_idx] >= 0 {
                            let neighbor = points[grid[cell_idx] as usize];
                            let dist_sq = (candidate - neighbor).length_squared();
                            if dist_sq < r * r {
                                too_close = true;
                                break 'check;
                            }
                        }
                    }
                }

                if !too_close {
                    let new_idx = points.len();
                    points.push(candidate);
                    active_list.push(new_idx);
                    if cx < grid_width && cy < grid_height {
                        grid[cy * grid_width + cx] = new_idx as i32;
                    }
                    found = true;
                    break;
                }
            }

            if !found {
                // Remove this point from the active list.
                active_list.swap_remove(active_idx);
            }
        }

        points
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn rng_different_seeds() {
        let mut rng1 = Rng::new(0);
        let mut rng2 = Rng::new(1);
        let v1 = rng1.next_u64();
        let v2 = rng2.next_u64();
        assert_ne!(v1, v2, "Different seeds should produce different values");
    }

    #[test]
    fn rng_f32_range() {
        let mut rng = Rng::new(42);
        for _ in 0..10000 {
            let val = rng.next_f32();
            assert!(val >= 0.0 && val < 1.0, "f32 out of [0, 1): {val}");
        }
    }

    #[test]
    fn rng_range() {
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let val = rng.range(5.0, 10.0);
            assert!(val >= 5.0 && val < 10.0, "range out of [5, 10): {val}");
        }
    }

    #[test]
    fn rng_range_i32() {
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let val = rng.range_i32(1, 7); // 1d6
            assert!(val >= 1 && val < 7, "range_i32 out of [1, 7): {val}");
        }
    }

    #[test]
    fn rng_range_i32_single_value() {
        let mut rng = Rng::new(42);
        let val = rng.range_i32(5, 5);
        assert_eq!(val, 5);
    }

    #[test]
    fn rng_bool_always_true() {
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            assert!(rng.bool(1.0));
        }
    }

    #[test]
    fn rng_bool_always_false() {
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            assert!(!rng.bool(0.0));
        }
    }

    #[test]
    fn rng_unit_circle_on_circle() {
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let p = rng.unit_circle();
            let len = p.length();
            assert!((len - 1.0).abs() < 0.01, "Point should be on unit circle, length={len}");
        }
    }

    #[test]
    fn rng_unit_sphere_on_sphere() {
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let p = rng.unit_sphere();
            let len = p.length();
            assert!((len - 1.0).abs() < 0.01, "Point should be on unit sphere, length={len}");
        }
    }

    #[test]
    fn rng_in_sphere() {
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let p = rng.in_sphere(5.0);
            assert!(p.length() <= 5.01, "Point should be inside sphere");
        }
    }

    #[test]
    fn rng_in_circle() {
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let p = rng.in_circle(3.0);
            assert!(p.length() <= 3.01, "Point should be inside circle");
        }
    }

    #[test]
    fn rng_shuffle() {
        let mut rng = Rng::new(42);
        let mut arr = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let original = arr.clone();
        rng.shuffle(&mut arr);
        // The shuffled array should contain the same elements.
        let mut sorted = arr.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        // It should (very likely) be in a different order.
        assert_ne!(arr, original, "Shuffle should change order (extremely unlikely to be identical)");
    }

    #[test]
    fn rng_pick() {
        let mut rng = Rng::new(42);
        let items = vec!["a", "b", "c", "d"];
        let picked = rng.pick(&items).unwrap();
        assert!(items.contains(picked));
    }

    #[test]
    fn rng_pick_empty() {
        let mut rng = Rng::new(42);
        let items: Vec<i32> = vec![];
        assert!(rng.pick(&items).is_none());
    }

    #[test]
    fn rng_weighted_pick() {
        let mut rng = Rng::new(42);
        let weights = vec![0.0, 0.0, 1.0, 0.0];
        // With only one non-zero weight, should always pick index 2.
        for _ in 0..100 {
            assert_eq!(rng.weighted_pick(&weights), 2);
        }
    }

    #[test]
    fn rng_weighted_pick_distribution() {
        let mut rng = Rng::new(42);
        let weights = vec![1.0, 3.0]; // 25% / 75% expected
        let mut counts = [0u32; 2];
        let n = 10000;
        for _ in 0..n {
            counts[rng.weighted_pick(&weights)] += 1;
        }
        let ratio = counts[1] as f32 / n as f32;
        assert!(ratio > 0.6 && ratio < 0.9, "Index 1 should be ~75%, got {}", ratio * 100.0);
    }

    #[test]
    fn rng_gaussian_mean() {
        let mut rng = Rng::new(42);
        let n = 10000;
        let sum: f32 = (0..n).map(|_| rng.gaussian()).sum();
        let mean = sum / n as f32;
        assert!(mean.abs() < 0.1, "Gaussian mean should be ~0, got {mean}");
    }

    #[test]
    fn rng_hemisphere() {
        let mut rng = Rng::new(42);
        let normal = Vec3::Y;
        for _ in 0..100 {
            let dir = rng.hemisphere(normal);
            assert!(dir.dot(normal) >= 0.0, "Hemisphere sample should be above the plane");
        }
    }

    #[test]
    fn rng_cosine_hemisphere() {
        let mut rng = Rng::new(42);
        let normal = Vec3::Y;
        for _ in 0..100 {
            let dir = rng.cosine_hemisphere(normal);
            assert!(dir.dot(normal) >= -0.01, "Cosine hemisphere sample should be above the plane");
            assert!((dir.length() - 1.0).abs() < 0.01, "Should be unit length");
        }
    }

    // -- Halton sequence tests --

    #[test]
    fn halton_first_values() {
        let mut h = Halton::new();
        let p1 = h.next();
        // First value of base-2 Halton: 1/2 = 0.5
        assert!((p1.x - 0.5).abs() < 0.01);
        // First value of base-3 Halton: 1/3 ≈ 0.333
        assert!((p1.y - 0.333).abs() < 0.02);
    }

    #[test]
    fn halton_range() {
        let mut h = Halton::new();
        for _ in 0..100 {
            let p = h.next();
            assert!(p.x >= 0.0 && p.x < 1.0, "Halton x out of range: {}", p.x);
            assert!(p.y >= 0.0 && p.y < 1.0, "Halton y out of range: {}", p.y);
        }
    }

    #[test]
    fn halton_sample_by_index() {
        let h = Halton::new();
        let p = h.sample(0);
        assert!((p.x - 0.5).abs() < 0.01);
    }

    #[test]
    fn halton_reset() {
        let mut h = Halton::new();
        let p1 = h.next();
        h.reset();
        let p2 = h.next();
        assert_eq!(p1, p2);
    }

    // -- PoissonDisk tests --

    #[test]
    fn poisson_disk_min_distance() {
        let mut rng = Rng::new(42);
        let sampler = PoissonDisk::new(2.0, 20.0, 20.0);
        let points = sampler.generate(&mut rng);

        assert!(!points.is_empty(), "Should generate at least one point");

        // Verify minimum distance between all pairs.
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dist = (points[i] - points[j]).length();
                assert!(
                    dist >= 1.99, // small epsilon for float precision
                    "Points {i} and {j} are too close: {dist} < 2.0"
                );
            }
        }
    }

    #[test]
    fn poisson_disk_bounds() {
        let mut rng = Rng::new(42);
        let sampler = PoissonDisk::new(1.0, 10.0, 10.0);
        let points = sampler.generate(&mut rng);

        for (i, p) in points.iter().enumerate() {
            assert!(
                p.x >= 0.0 && p.x < 10.0 && p.y >= 0.0 && p.y < 10.0,
                "Point {i} out of bounds: {p:?}"
            );
        }
    }

    #[test]
    fn poisson_disk_deterministic() {
        let sampler = PoissonDisk::new(1.5, 10.0, 10.0);
        let mut rng1 = Rng::new(42);
        let points1 = sampler.generate(&mut rng1);
        let mut rng2 = Rng::new(42);
        let points2 = sampler.generate(&mut rng2);
        assert_eq!(points1.len(), points2.len());
        for (a, b) in points1.iter().zip(points2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn poisson_disk_reasonable_count() {
        let mut rng = Rng::new(42);
        let sampler = PoissonDisk::new(1.0, 10.0, 10.0);
        let points = sampler.generate(&mut rng);
        // A 10x10 area with r=1 should fit roughly 60-120 points.
        assert!(
            points.len() > 30 && points.len() < 200,
            "Unexpected point count: {}",
            points.len()
        );
    }
}
