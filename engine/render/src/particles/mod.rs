// engine/render/src/particles/mod.rs
//
// Particle system module for the Genovo engine.
//
// Provides a high-performance, data-oriented particle system with support for
// multiple emitter shapes, force fields, collision detection, and rendering
// modes. The system is designed around Structure-of-Arrays (SoA) layout for
// cache-friendly simulation of thousands of particles per frame.
//
// # Architecture
//
// - [`emitter`] -- Particle emitters with configurable shapes and emission modes.
// - [`particle`] -- SoA particle pool, integration, and lifetime management.
// - [`forces`] -- Pluggable force fields (gravity, wind, vortex, turbulence).
// - [`collision`] -- Particle-geometry collision detection and response.
// - [`renderer`] -- Billboard, stretched billboard, mesh, and trail rendering.
// - [`system`] -- High-level particle system component and manager.

pub mod emitter;
pub mod particle;
pub mod forces;
pub mod collision;
pub mod renderer;
pub mod system;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use emitter::{
    EmissionMode, EmitterShape, ParticleEmitter, SimulationSpace,
};
pub use particle::{ParticlePool, SortMode};
pub use forces::{
    AttractorForce, CurlNoise, DragForce, ForceField, GravityForce, NoiseField,
    TurbulenceForce, VortexForce, WindForce,
};
pub use collision::{CollisionSettings, ParticleCollider};
pub use renderer::{
    ParticleRenderMode, ParticleVertex, SpriteSheet, SoftParticleSettings,
};
pub use system::{ParticleSystem, ParticleSystemManager, PlaybackState};

// ---------------------------------------------------------------------------
// Core types used across sub-modules
// ---------------------------------------------------------------------------

use glam::{Vec2, Vec3};

/// A color gradient defined by a series of color stops.
///
/// Colors are linearly interpolated between stops. The gradient parameter
/// `t` is expected in `[0, 1]`.
#[derive(Debug, Clone)]
pub struct ColorGradient {
    /// Sorted by `t` ascending. Each entry is `(t, [r, g, b, a])`.
    pub stops: Vec<(f32, [f32; 4])>,
}

impl ColorGradient {
    /// Creates a simple two-stop gradient from `start` to `end`.
    pub fn new(start: [f32; 4], end: [f32; 4]) -> Self {
        Self {
            stops: vec![(0.0, start), (1.0, end)],
        }
    }

    /// Creates a gradient with multiple stops.
    pub fn from_stops(stops: Vec<(f32, [f32; 4])>) -> Self {
        let mut s = stops;
        s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { stops: s }
    }

    /// Evaluates the gradient at parameter `t` (clamped to [0, 1]).
    pub fn evaluate(&self, t: f32) -> [f32; 4] {
        if self.stops.is_empty() {
            return [1.0, 1.0, 1.0, 1.0];
        }
        let t = t.clamp(0.0, 1.0);

        if t <= self.stops[0].0 {
            return self.stops[0].1;
        }
        if t >= self.stops[self.stops.len() - 1].0 {
            return self.stops[self.stops.len() - 1].1;
        }

        // Find the two surrounding stops.
        for i in 0..self.stops.len() - 1 {
            let (t0, c0) = &self.stops[i];
            let (t1, c1) = &self.stops[i + 1];
            if t >= *t0 && t <= *t1 {
                let frac = if (t1 - t0).abs() < 1e-9 {
                    0.0
                } else {
                    (t - t0) / (t1 - t0)
                };
                return [
                    c0[0] + (c1[0] - c0[0]) * frac,
                    c0[1] + (c1[1] - c0[1]) * frac,
                    c0[2] + (c1[2] - c0[2]) * frac,
                    c0[3] + (c1[3] - c0[3]) * frac,
                ];
            }
        }
        self.stops[self.stops.len() - 1].1
    }
}

impl Default for ColorGradient {
    fn default() -> Self {
        Self::new([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0])
    }
}

/// A curve defined by keyframes, used for size-over-lifetime, speed-over-lifetime, etc.
///
/// The parameter `t` is expected in `[0, 1]`.
#[derive(Debug, Clone)]
pub struct Curve {
    /// Sorted by `t` ascending. Each entry is `(t, value)`.
    pub keys: Vec<(f32, f32)>,
}

impl Curve {
    /// Creates a constant curve.
    pub fn constant(value: f32) -> Self {
        Self {
            keys: vec![(0.0, value), (1.0, value)],
        }
    }

    /// Creates a linear curve from `start` to `end`.
    pub fn linear(start: f32, end: f32) -> Self {
        Self {
            keys: vec![(0.0, start), (1.0, end)],
        }
    }

    /// Creates a curve from keyframes.
    pub fn from_keys(keys: Vec<(f32, f32)>) -> Self {
        let mut k = keys;
        k.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { keys: k }
    }

    /// Evaluates the curve at parameter `t` (clamped to [0, 1]).
    pub fn evaluate(&self, t: f32) -> f32 {
        if self.keys.is_empty() {
            return 1.0;
        }
        let t = t.clamp(0.0, 1.0);

        if t <= self.keys[0].0 {
            return self.keys[0].1;
        }
        if t >= self.keys[self.keys.len() - 1].0 {
            return self.keys[self.keys.len() - 1].1;
        }

        for i in 0..self.keys.len() - 1 {
            let (t0, v0) = self.keys[i];
            let (t1, v1) = self.keys[i + 1];
            if t >= t0 && t <= t1 {
                let frac = if (t1 - t0).abs() < 1e-9 {
                    0.0
                } else {
                    (t - t0) / (t1 - t0)
                };
                return v0 + (v1 - v0) * frac;
            }
        }
        self.keys[self.keys.len() - 1].1
    }
}

impl Default for Curve {
    fn default() -> Self {
        Self::constant(1.0)
    }
}

/// A simple pseudo-random number generator using xorshift64.
///
/// This is intentionally not cryptographically secure. It is fast, small,
/// and deterministic -- ideal for particle spawning where we need many
/// cheap random numbers per frame and reproducibility for replays.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Returns the next u64 value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a random f32 in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0x00FF_FFFF) as f32 / 16_777_216.0
    }

    /// Returns a random f32 in `[-1, 1)`.
    #[inline]
    pub fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// Returns a random f32 in `[min, max)`.
    #[inline]
    pub fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }

    /// Returns a random unit vector on the surface of a sphere.
    pub fn unit_sphere(&mut self) -> Vec3 {
        // Marsaglia's method: pick from unit cube, reject outside sphere.
        loop {
            let x = self.next_f32_signed();
            let y = self.next_f32_signed();
            let z = self.next_f32_signed();
            let len_sq = x * x + y * y + z * z;
            if len_sq > 1e-9 && len_sq <= 1.0 {
                let inv_len = 1.0 / len_sq.sqrt();
                return Vec3::new(x * inv_len, y * inv_len, z * inv_len);
            }
        }
    }

    /// Returns a random point inside a unit sphere.
    pub fn inside_unit_sphere(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(
                self.next_f32_signed(),
                self.next_f32_signed(),
                self.next_f32_signed(),
            );
            if v.length_squared() <= 1.0 {
                return v;
            }
        }
    }

    /// Returns a random point inside a unit circle (XY plane).
    pub fn inside_unit_circle(&mut self) -> Vec2 {
        loop {
            let v = Vec2::new(self.next_f32_signed(), self.next_f32_signed());
            if v.length_squared() <= 1.0 {
                return v;
            }
        }
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(0xDEAD_BEEF_CAFE_BABE)
    }
}
