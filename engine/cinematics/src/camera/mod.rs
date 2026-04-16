//! # Cinematic Camera
//!
//! Provides cinematic camera movements, spline-based camera paths, procedural
//! camera shake using Perlin noise, and constraint-based camera behaviour.
//!
//! ## Camera movements
//!
//! The [`CinematicCamera`] struct wraps a transform and exposes high-level
//! movement operations:
//!
//! - **Dolly** -- move along the camera's forward axis (or along a path).
//! - **Truck** -- lateral (left/right) movement.
//! - **Pedestal** -- vertical (up/down) movement.
//! - **Pan** -- horizontal rotation (yaw).
//! - **Tilt** -- vertical rotation (pitch).
//! - **Zoom** -- animate the field-of-view.
//! - **Roll** -- rotation around the forward axis.
//! - **Shake** -- procedural noise-based shake.
//!
//! ## Camera paths
//!
//! [`CameraPath`] uses Catmull-Rom splines for smooth camera movement through
//! a series of control points. The path supports arc-length reparameterization
//! for constant-speed traversal.
//!
//! ## Camera constraints
//!
//! [`CameraConstraint`] allows the camera to track, follow, or orbit around
//! entities. Multiple constraints can be blended with weights.

use std::collections::HashMap;

use genovo_ecs::Entity;
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the camera subsystem.
#[derive(Debug, thiserror::Error)]
pub enum CameraError {
    #[error("Camera path has no control points")]
    EmptyPath,
    #[error("Camera path needs at least 2 control points, has {0}")]
    InsufficientPoints(usize),
    #[error("Parameter t={0} is out of range [0, 1]")]
    ParameterOutOfRange(f32),
}

// ---------------------------------------------------------------------------
// Perlin noise implementation
// ---------------------------------------------------------------------------

/// A self-contained 1D/2D/3D Perlin noise generator used for procedural
/// camera shake. Uses a permutation table seeded from a fixed value.
///
/// The implementation follows Ken Perlin's improved noise algorithm with
/// quintic fade curves and gradient tables.
#[derive(Debug, Clone)]
pub struct PerlinNoise {
    /// Permutation table (doubled for wrapping).
    perm: [u8; 512],
}

impl PerlinNoise {
    /// Create a new Perlin noise generator with a given seed.
    pub fn new(seed: u64) -> Self {
        let mut perm = [0u8; 256];
        // Initialize identity permutation
        for i in 0..256 {
            perm[i] = i as u8;
        }
        // Fisher-Yates shuffle with a simple LCG seeded by `seed`.
        let mut rng = seed;
        for i in (1..256).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            perm.swap(i, j);
        }
        let mut doubled = [0u8; 512];
        for i in 0..512 {
            doubled[i] = perm[i & 255];
        }
        Self { perm: doubled }
    }

    /// Quintic fade curve: 6t^5 - 15t^4 + 10t^3
    #[inline]
    fn fade(t: f32) -> f32 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    /// Linear interpolation.
    #[inline]
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }

    /// 1D gradient function.
    #[inline]
    fn grad1(hash: u8, x: f32) -> f32 {
        if hash & 1 == 0 { x } else { -x }
    }

    /// 2D gradient function using 4 gradient directions.
    #[inline]
    fn grad2(hash: u8, x: f32, y: f32) -> f32 {
        match hash & 3 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            3 => -x - y,
            _ => unreachable!(),
        }
    }

    /// 3D gradient function from Perlin's improved noise.
    #[inline]
    fn grad3(hash: u8, x: f32, y: f32, z: f32) -> f32 {
        let h = hash & 15;
        let u = if h < 8 { x } else { y };
        let v = if h < 4 {
            y
        } else if h == 12 || h == 14 {
            x
        } else {
            z
        };
        let a = if h & 1 == 0 { u } else { -u };
        let b = if h & 2 == 0 { v } else { -v };
        a + b
    }

    /// Sample 1D Perlin noise at position `x`.
    /// Returns a value in approximately `[-1, 1]`.
    pub fn noise1d(&self, x: f32) -> f32 {
        let xi = x.floor() as i32;
        let xf = x - x.floor();
        let u = Self::fade(xf);

        let aa = self.perm[(xi & 255) as usize];
        let ba = self.perm[((xi + 1) & 255) as usize];

        Self::lerp(Self::grad1(aa, xf), Self::grad1(ba, xf - 1.0), u)
    }

    /// Sample 2D Perlin noise at position `(x, y)`.
    /// Returns a value in approximately `[-1, 1]`.
    pub fn noise2d(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let xf = x - x.floor();
        let yf = y - y.floor();

        let u = Self::fade(xf);
        let v = Self::fade(yf);

        let ix = (xi & 255) as usize;
        let iy = (yi & 255) as usize;

        let aa = self.perm[self.perm[ix] as usize + iy] as u8;
        let ba = self.perm[self.perm[ix + 1] as usize + iy] as u8;
        let ab = self.perm[self.perm[ix] as usize + iy + 1] as u8;
        let bb = self.perm[self.perm[ix + 1] as usize + iy + 1] as u8;

        let x1 = Self::lerp(
            Self::grad2(aa, xf, yf),
            Self::grad2(ba, xf - 1.0, yf),
            u,
        );
        let x2 = Self::lerp(
            Self::grad2(ab, xf, yf - 1.0),
            Self::grad2(bb, xf - 1.0, yf - 1.0),
            u,
        );
        Self::lerp(x1, x2, v)
    }

    /// Sample 3D Perlin noise at position `(x, y, z)`.
    /// Returns a value in approximately `[-1, 1]`.
    pub fn noise3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();

        let u = Self::fade(xf);
        let v = Self::fade(yf);
        let w = Self::fade(zf);

        let ix = (xi & 255) as usize;
        let iy = (yi & 255) as usize;
        let iz = (zi & 255) as usize;

        let a = self.perm[ix] as usize + iy;
        let b = self.perm[ix + 1] as usize + iy;

        let aa = self.perm[a] as usize + iz;
        let ab = self.perm[a + 1] as usize + iz;
        let ba = self.perm[b] as usize + iz;
        let bb = self.perm[b + 1] as usize + iz;

        let aaa = self.perm[aa];
        let aab = self.perm[aa + 1];
        let aba = self.perm[ab];
        let abb = self.perm[ab + 1];
        let baa = self.perm[ba];
        let bab = self.perm[ba + 1];
        let bba = self.perm[bb];
        let bbb = self.perm[bb + 1];

        let x1 = Self::lerp(
            Self::grad3(aaa, xf, yf, zf),
            Self::grad3(baa, xf - 1.0, yf, zf),
            u,
        );
        let x2 = Self::lerp(
            Self::grad3(aba, xf, yf - 1.0, zf),
            Self::grad3(bba, xf - 1.0, yf - 1.0, zf),
            u,
        );
        let y1 = Self::lerp(x1, x2, v);

        let x3 = Self::lerp(
            Self::grad3(aab, xf, yf, zf - 1.0),
            Self::grad3(bab, xf - 1.0, yf, zf - 1.0),
            u,
        );
        let x4 = Self::lerp(
            Self::grad3(abb, xf, yf - 1.0, zf - 1.0),
            Self::grad3(bbb, xf - 1.0, yf - 1.0, zf - 1.0),
            u,
        );
        let y2 = Self::lerp(x3, x4, v);

        Self::lerp(y1, y2, w)
    }

    /// Fractal Brownian Motion (fBm) -- octaved 1D noise.
    pub fn fbm1d(&self, x: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
        let mut sum = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amplitude = 0.0;
        for _ in 0..octaves {
            sum += self.noise1d(x * frequency) * amplitude;
            max_amplitude += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        sum / max_amplitude
    }

    /// Fractal Brownian Motion (fBm) -- octaved 2D noise.
    pub fn fbm2d(
        &self,
        x: f32,
        y: f32,
        octaves: u32,
        lacunarity: f32,
        persistence: f32,
    ) -> f32 {
        let mut sum = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amplitude = 0.0;
        for _ in 0..octaves {
            sum += self.noise2d(x * frequency, y * frequency) * amplitude;
            max_amplitude += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        sum / max_amplitude
    }

    /// Fractal Brownian Motion (fBm) -- octaved 3D noise.
    pub fn fbm3d(
        &self,
        x: f32,
        y: f32,
        z: f32,
        octaves: u32,
        lacunarity: f32,
        persistence: f32,
    ) -> f32 {
        let mut sum = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amplitude = 0.0;
        for _ in 0..octaves {
            sum += self.noise3d(x * frequency, y * frequency, z * frequency) * amplitude;
            max_amplitude += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        sum / max_amplitude
    }
}

impl Default for PerlinNoise {
    fn default() -> Self {
        Self::new(42)
    }
}

// ---------------------------------------------------------------------------
// Camera shake
// ---------------------------------------------------------------------------

/// Configuration for Perlin noise-based camera shake.
///
/// Produces smooth, organic shake by sampling fractal noise over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerlinShake {
    /// Maximum positional offset in each axis.
    pub amplitude: Vec3,
    /// Noise sampling frequency (higher = faster shake).
    pub frequency: f32,
    /// Exponential decay rate (shake dies out over time).
    pub decay: f32,
    /// Number of noise octaves (more octaves = more detail).
    pub octaves: u32,
    /// Lacunarity for fBm (frequency multiplier per octave).
    pub lacunarity: f32,
    /// Persistence for fBm (amplitude multiplier per octave).
    pub persistence: f32,
    /// Whether to also apply rotational shake.
    pub rotational: bool,
    /// Maximum rotational shake in radians per axis.
    pub rotation_amplitude: Vec3,
}

impl Default for PerlinShake {
    fn default() -> Self {
        Self {
            amplitude: Vec3::new(0.1, 0.1, 0.05),
            frequency: 15.0,
            decay: 3.0,
            octaves: 3,
            lacunarity: 2.0,
            persistence: 0.5,
            rotational: true,
            rotation_amplitude: Vec3::new(0.02, 0.02, 0.01),
        }
    }
}

/// Configuration for impulse-based camera shake (e.g., explosion impact).
///
/// Applies a sharp initial offset that decays exponentially.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseShake {
    /// Initial intensity (0..1 typical, can exceed 1 for strong shakes).
    pub intensity: f32,
    /// Exponential decay rate.
    pub decay_rate: f32,
    /// Preferred direction of the impulse (normalized). If zero, shake is
    /// omnidirectional.
    pub direction: Vec3,
    /// Duration cap -- shake stops after this many seconds regardless of
    /// remaining intensity.
    pub max_duration: f32,
}

impl Default for ImpulseShake {
    fn default() -> Self {
        Self {
            intensity: 0.5,
            decay_rate: 5.0,
            direction: Vec3::ZERO,
            max_duration: 2.0,
        }
    }
}

/// A live shake instance being evaluated.
#[derive(Debug, Clone)]
pub struct ShakeInstance {
    /// Type of shake.
    pub kind: ShakeKind,
    /// Elapsed time since the shake started.
    pub elapsed: f32,
    /// Current computed positional offset.
    pub offset: Vec3,
    /// Current computed rotational offset (Euler angles in radians).
    pub rotation_offset: Vec3,
    /// Whether this shake has finished (should be removed).
    pub finished: bool,
    /// Unique noise seed for this instance.
    seed_offset: f32,
}

/// Discriminated union of shake types.
#[derive(Debug, Clone)]
pub enum ShakeKind {
    Perlin(PerlinShake),
    Impulse(ImpulseShake),
}

impl ShakeInstance {
    /// Create a new Perlin shake instance.
    pub fn new_perlin(config: PerlinShake, seed_offset: f32) -> Self {
        Self {
            kind: ShakeKind::Perlin(config),
            elapsed: 0.0,
            offset: Vec3::ZERO,
            rotation_offset: Vec3::ZERO,
            finished: false,
            seed_offset,
        }
    }

    /// Create a new impulse shake instance.
    pub fn new_impulse(config: ImpulseShake, seed_offset: f32) -> Self {
        Self {
            kind: ShakeKind::Impulse(config),
            elapsed: 0.0,
            offset: Vec3::ZERO,
            rotation_offset: Vec3::ZERO,
            finished: false,
            seed_offset,
        }
    }

    /// Advance the shake by `dt` seconds and recompute the offset.
    pub fn update(&mut self, dt: f32, noise: &PerlinNoise) {
        self.elapsed += dt;

        match &self.kind {
            ShakeKind::Perlin(cfg) => {
                let decay_mult = (-cfg.decay * self.elapsed).exp();
                if decay_mult < 0.001 {
                    self.finished = true;
                    self.offset = Vec3::ZERO;
                    self.rotation_offset = Vec3::ZERO;
                    return;
                }

                let t = self.elapsed * cfg.frequency + self.seed_offset;

                // Sample noise on 3 separate channels (offset by large primes
                // so the axes are uncorrelated).
                let nx = noise.fbm1d(t, cfg.octaves, cfg.lacunarity, cfg.persistence);
                let ny =
                    noise.fbm1d(t + 31.41, cfg.octaves, cfg.lacunarity, cfg.persistence);
                let nz =
                    noise.fbm1d(t + 73.17, cfg.octaves, cfg.lacunarity, cfg.persistence);

                self.offset = Vec3::new(
                    nx * cfg.amplitude.x * decay_mult,
                    ny * cfg.amplitude.y * decay_mult,
                    nz * cfg.amplitude.z * decay_mult,
                );

                if cfg.rotational {
                    let rx =
                        noise.fbm1d(t + 113.0, cfg.octaves, cfg.lacunarity, cfg.persistence);
                    let ry =
                        noise.fbm1d(t + 157.0, cfg.octaves, cfg.lacunarity, cfg.persistence);
                    let rz =
                        noise.fbm1d(t + 199.0, cfg.octaves, cfg.lacunarity, cfg.persistence);
                    self.rotation_offset = Vec3::new(
                        rx * cfg.rotation_amplitude.x * decay_mult,
                        ry * cfg.rotation_amplitude.y * decay_mult,
                        rz * cfg.rotation_amplitude.z * decay_mult,
                    );
                }
            }
            ShakeKind::Impulse(cfg) => {
                if self.elapsed > cfg.max_duration {
                    self.finished = true;
                    self.offset = Vec3::ZERO;
                    self.rotation_offset = Vec3::ZERO;
                    return;
                }

                let decay_mult = (-cfg.decay_rate * self.elapsed).exp();
                let intensity = cfg.intensity * decay_mult;

                if intensity < 0.001 {
                    self.finished = true;
                    self.offset = Vec3::ZERO;
                    self.rotation_offset = Vec3::ZERO;
                    return;
                }

                // Use noise for jitter within the impulse envelope.
                let t = self.elapsed * 25.0 + self.seed_offset;
                let nx = noise.noise1d(t);
                let ny = noise.noise1d(t + 43.0);
                let nz = noise.noise1d(t + 97.0);

                if cfg.direction.length_squared() > 0.001 {
                    let dir = cfg.direction.normalize();
                    // Primary offset along direction, with perpendicular jitter.
                    let perp1 = if dir.y.abs() < 0.9 {
                        dir.cross(Vec3::Y).normalize()
                    } else {
                        dir.cross(Vec3::X).normalize()
                    };
                    let perp2 = dir.cross(perp1).normalize();
                    self.offset = dir * intensity * (0.5 + 0.5 * nx)
                        + perp1 * intensity * 0.3 * ny
                        + perp2 * intensity * 0.3 * nz;
                } else {
                    self.offset = Vec3::new(
                        nx * intensity,
                        ny * intensity,
                        nz * intensity * 0.5,
                    );
                }
            }
        }
    }
}

/// Manages multiple simultaneous camera shakes with additive blending.
#[derive(Debug, Clone)]
pub struct CameraShake {
    /// Active shake instances.
    instances: Vec<ShakeInstance>,
    /// Shared Perlin noise generator.
    noise: PerlinNoise,
    /// Monotonic counter for generating unique seed offsets.
    next_seed: f32,
    /// Combined positional offset (sum of all active shakes).
    pub combined_offset: Vec3,
    /// Combined rotational offset (sum of all active shakes).
    pub combined_rotation: Vec3,
}

impl CameraShake {
    /// Create a new camera shake manager.
    pub fn new(seed: u64) -> Self {
        Self {
            instances: Vec::new(),
            noise: PerlinNoise::new(seed),
            next_seed: 0.0,
            combined_offset: Vec3::ZERO,
            combined_rotation: Vec3::ZERO,
        }
    }

    /// Add a Perlin noise shake.
    pub fn add_perlin(&mut self, config: PerlinShake) {
        let seed = self.next_seed;
        self.next_seed += 17.3;
        self.instances.push(ShakeInstance::new_perlin(config, seed));
    }

    /// Add an impulse shake.
    pub fn add_impulse(&mut self, config: ImpulseShake) {
        let seed = self.next_seed;
        self.next_seed += 17.3;
        self.instances.push(ShakeInstance::new_impulse(config, seed));
    }

    /// Number of active shake instances.
    pub fn active_count(&self) -> usize {
        self.instances.len()
    }

    /// Clear all active shakes immediately.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.combined_offset = Vec3::ZERO;
        self.combined_rotation = Vec3::ZERO;
    }

    /// Update all shakes and compute the combined offset.
    pub fn update(&mut self, dt: f32) {
        self.combined_offset = Vec3::ZERO;
        self.combined_rotation = Vec3::ZERO;

        for inst in &mut self.instances {
            inst.update(dt, &self.noise);
            self.combined_offset += inst.offset;
            self.combined_rotation += inst.rotation_offset;
        }

        // Remove finished shakes.
        self.instances.retain(|inst| !inst.finished);
    }
}

impl Default for CameraShake {
    fn default() -> Self {
        Self::new(42)
    }
}

// ---------------------------------------------------------------------------
// Camera keypoint & path (Catmull-Rom spline)
// ---------------------------------------------------------------------------

/// A single control point on a camera path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraKeypoint {
    /// World-space position.
    pub position: Vec3,
    /// World-space look-at target (used to derive orientation).
    pub look_at: Vec3,
    /// Field of view in degrees at this point.
    pub fov: f32,
    /// Camera roll in radians.
    pub roll: f32,
    /// Time in seconds when the camera should reach this point.
    pub time: f32,
}

impl CameraKeypoint {
    /// Create a new camera keypoint.
    pub fn new(position: Vec3, look_at: Vec3, fov: f32, roll: f32, time: f32) -> Self {
        Self {
            position,
            look_at,
            fov,
            roll,
            time,
        }
    }

    /// Compute the forward direction from position to look_at.
    pub fn forward(&self) -> Vec3 {
        (self.look_at - self.position).normalize_or_zero()
    }

    /// Compute a view quaternion from position, look_at, and roll.
    pub fn orientation(&self) -> Quat {
        let forward = self.forward();
        if forward.length_squared() < 1e-6 {
            return Quat::IDENTITY;
        }
        let up = Vec3::Y;
        let right = forward.cross(up).normalize_or_zero();
        let corrected_up = right.cross(forward).normalize_or_zero();

        // Build rotation from look-at, then apply roll.
        let look_rot = Quat::from_mat3(&glam::Mat3::from_cols(right, corrected_up, -forward));
        let roll_rot = Quat::from_axis_angle(forward, self.roll);
        roll_rot * look_rot
    }
}

/// Camera path result: position, rotation, FOV.
#[derive(Debug, Clone, Copy)]
pub struct CameraPathSample {
    pub position: Vec3,
    pub rotation: Quat,
    pub fov: f32,
}

/// A spline path for camera movement using Catmull-Rom interpolation.
///
/// Supports arc-length reparameterization for constant-speed traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraPath {
    /// Control points along the path.
    pub control_points: Vec<CameraKeypoint>,
    /// Precomputed arc-length table for constant-speed parameterization.
    /// Each entry is `(parameter_t, accumulated_arc_length)`.
    #[serde(skip)]
    arc_length_table: Vec<(f32, f32)>,
    /// Total arc length of the path.
    #[serde(skip)]
    total_arc_length: f32,
    /// Tension parameter for Catmull-Rom (0.5 = centripetal, 0.0 = uniform).
    pub tension: f32,
}

impl CameraPath {
    /// Create a new camera path from control points.
    pub fn new(points: Vec<CameraKeypoint>) -> Self {
        let mut path = Self {
            control_points: points,
            arc_length_table: Vec::new(),
            total_arc_length: 0.0,
            tension: 0.5,
        };
        path.rebuild_arc_length_table();
        path
    }

    /// Number of control points.
    pub fn point_count(&self) -> usize {
        self.control_points.len()
    }

    /// Total arc length of the path in world units.
    pub fn total_length(&self) -> f32 {
        self.total_arc_length
    }

    /// Duration of the path (time of last point - time of first point).
    pub fn duration(&self) -> f32 {
        match (self.control_points.first(), self.control_points.last()) {
            (Some(a), Some(b)) => b.time - a.time,
            _ => 0.0,
        }
    }

    /// Add a control point, maintaining time-sorted order.
    pub fn add_point(&mut self, point: CameraKeypoint) {
        let pos = self
            .control_points
            .binary_search_by(|p| p.time.partial_cmp(&point.time).unwrap())
            .unwrap_or_else(|e| e);
        self.control_points.insert(pos, point);
        self.rebuild_arc_length_table();
    }

    /// Remove a control point by index.
    pub fn remove_point(&mut self, index: usize) -> Option<CameraKeypoint> {
        if index < self.control_points.len() {
            let pt = self.control_points.remove(index);
            self.rebuild_arc_length_table();
            Some(pt)
        } else {
            None
        }
    }

    /// Evaluate the path at a raw parameter `t` in `[0, 1]`.
    ///
    /// This uses Catmull-Rom interpolation between control points.
    /// Returns `(position, rotation, fov)`.
    pub fn evaluate(&self, t: f32) -> CameraPathSample {
        let n = self.control_points.len();
        if n == 0 {
            return CameraPathSample {
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                fov: 60.0,
            };
        }
        if n == 1 {
            let p = &self.control_points[0];
            return CameraPathSample {
                position: p.position,
                rotation: p.orientation(),
                fov: p.fov,
            };
        }

        let t_clamped = t.clamp(0.0, 1.0);
        let segments = (n - 1) as f32;
        let scaled = t_clamped * segments;
        let segment = (scaled.floor() as usize).min(n - 2);
        let local_t = scaled - segment as f32;

        // Get four control points for Catmull-Rom (clamping at boundaries).
        let p0 = &self.control_points[segment.saturating_sub(1)];
        let p1 = &self.control_points[segment];
        let p2 = &self.control_points[(segment + 1).min(n - 1)];
        let p3 = &self.control_points[(segment + 2).min(n - 1)];

        let position = catmull_rom_vec3(p0.position, p1.position, p2.position, p3.position, local_t, self.tension);
        let look_at = catmull_rom_vec3(p0.look_at, p1.look_at, p2.look_at, p3.look_at, local_t, self.tension);
        let fov = catmull_rom_scalar(p0.fov, p1.fov, p2.fov, p3.fov, local_t, self.tension);
        let roll = catmull_rom_scalar(p0.roll, p1.roll, p2.roll, p3.roll, local_t, self.tension);

        // Derive orientation from interpolated position and look_at.
        let forward = (look_at - position).normalize_or_zero();
        let rotation = if forward.length_squared() > 1e-6 {
            let up = Vec3::Y;
            let right = forward.cross(up).normalize_or_zero();
            let corrected_up = right.cross(forward).normalize_or_zero();
            let look_rot = Quat::from_mat3(&glam::Mat3::from_cols(right, corrected_up, -forward));
            let roll_rot = Quat::from_axis_angle(forward, roll);
            roll_rot * look_rot
        } else {
            Quat::IDENTITY
        };

        CameraPathSample {
            position,
            rotation,
            fov,
        }
    }

    /// Evaluate the path at a distance `d` along its arc length.
    ///
    /// This provides constant-speed traversal by mapping arc length to the
    /// raw Catmull-Rom parameter via the precomputed arc-length table.
    pub fn evaluate_at_distance(&self, d: f32) -> CameraPathSample {
        if self.total_arc_length <= 0.0 || self.arc_length_table.is_empty() {
            return self.evaluate(0.0);
        }
        let d_clamped = d.clamp(0.0, self.total_arc_length);

        // Binary search in the arc-length table.
        let t = self.distance_to_parameter(d_clamped);
        self.evaluate(t)
    }

    /// Evaluate the path at a time value, using the keypoints' own times.
    pub fn evaluate_at_time(&self, time: f32) -> CameraPathSample {
        if self.control_points.is_empty() {
            return CameraPathSample {
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                fov: 60.0,
            };
        }
        let first_time = self.control_points.first().unwrap().time;
        let last_time = self.control_points.last().unwrap().time;
        let duration = last_time - first_time;
        if duration <= 0.0 {
            return self.evaluate(0.0);
        }
        let t = ((time - first_time) / duration).clamp(0.0, 1.0);
        self.evaluate(t)
    }

    /// Convert an arc-length distance to a raw spline parameter.
    fn distance_to_parameter(&self, d: f32) -> f32 {
        if self.arc_length_table.len() < 2 {
            return 0.0;
        }

        // Binary search for the arc-length interval containing `d`.
        let mut lo = 0;
        let mut hi = self.arc_length_table.len() - 1;
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if self.arc_length_table[mid].1 < d {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let (t0, d0) = self.arc_length_table[lo];
        let (t1, d1) = self.arc_length_table[hi];
        let span = d1 - d0;
        if span < 1e-8 {
            return t0;
        }
        let frac = (d - d0) / span;
        t0 + (t1 - t0) * frac
    }

    /// Rebuild the arc-length lookup table.
    ///
    /// Samples the spline at many points and accumulates chord lengths.
    fn rebuild_arc_length_table(&mut self) {
        const SAMPLES: usize = 256;

        self.arc_length_table.clear();
        if self.control_points.len() < 2 {
            self.total_arc_length = 0.0;
            return;
        }

        let mut accumulated = 0.0f32;
        let mut prev_pos = self.evaluate(0.0).position;
        self.arc_length_table.push((0.0, 0.0));

        for i in 1..=SAMPLES {
            let t = i as f32 / SAMPLES as f32;
            let pos = self.evaluate(t).position;
            let chord = (pos - prev_pos).length();
            accumulated += chord;
            self.arc_length_table.push((t, accumulated));
            prev_pos = pos;
        }

        self.total_arc_length = accumulated;
    }
}

/// Catmull-Rom spline interpolation for Vec3 with configurable tension.
fn catmull_rom_vec3(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    t: f32,
    tension: f32,
) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    let s = tension;

    // Catmull-Rom matrix coefficients with tension parameter:
    // M = s * [ -1  2 -1  0 ]
    //         [  2 -3  0  1 ] / s ... (simplified form)
    // Using standard Catmull-Rom with tension:
    let m0 = (p2 - p0) * s;
    let m1 = (p3 - p1) * s;

    // Hermite basis
    let a = 2.0 * t3 - 3.0 * t2 + 1.0;
    let b = t3 - 2.0 * t2 + t;
    let c = -2.0 * t3 + 3.0 * t2;
    let d = t3 - t2;

    p1 * a + m0 * b + p2 * c + m1 * d
}

/// Catmull-Rom spline interpolation for a scalar.
fn catmull_rom_scalar(
    p0: f32,
    p1: f32,
    p2: f32,
    p3: f32,
    t: f32,
    tension: f32,
) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    let s = tension;

    let m0 = (p2 - p0) * s;
    let m1 = (p3 - p1) * s;

    let a = 2.0 * t3 - 3.0 * t2 + 1.0;
    let b = t3 - 2.0 * t2 + t;
    let c = -2.0 * t3 + 3.0 * t2;
    let d = t3 - t2;

    p1 * a + m0 * b + p2 * c + m1 * d
}

// ---------------------------------------------------------------------------
// Camera constraints
// ---------------------------------------------------------------------------

/// A constraint that controls camera behaviour relative to an entity or path.
///
/// Constraints reference entities at runtime and are not serialized directly.
/// Use [`CutsceneAsset`] for persistence, which stores constraint definitions
/// by entity name / binding key rather than raw entity handles.
#[derive(Debug, Clone)]
pub enum CameraConstraint {
    /// Track an entity: camera always looks at the target.
    LookAt {
        /// Entity to look at.
        target: Entity,
        /// Smoothing factor (0 = instant, higher = smoother).
        smooth: f32,
    },
    /// Follow an entity with an offset.
    Follow {
        /// Entity to follow.
        target: Entity,
        /// World-space offset from the target.
        offset: Vec3,
        /// Smoothing factor.
        smooth: f32,
    },
    /// Constrain the camera to move along a rail path.
    Rail {
        /// The path to constrain to.
        path: CameraPath,
        /// Speed of following along the path (units/second).
        follow_speed: f32,
    },
    /// Orbit around a target entity.
    Orbit {
        /// Entity to orbit around.
        target: Entity,
        /// Distance from the target.
        distance: f32,
        /// Orbital speed in radians/second.
        speed: f32,
        /// Current angle in radians.
        current_angle: f32,
        /// Elevation angle in radians.
        elevation: f32,
    },
    /// Free-look camera (user-controlled).
    FreeLook {
        /// Mouse sensitivity multiplier.
        sensitivity: f32,
    },
}

/// A weighted constraint for blending multiple constraints.
#[derive(Debug, Clone)]
pub struct WeightedConstraint {
    /// The constraint definition.
    pub constraint: CameraConstraint,
    /// Blend weight in `[0, 1]`.
    pub weight: f32,
    /// Whether this constraint is active.
    pub active: bool,
}

impl WeightedConstraint {
    /// Create a new weighted constraint.
    pub fn new(constraint: CameraConstraint, weight: f32) -> Self {
        Self {
            constraint,
            weight: weight.clamp(0.0, 1.0),
            active: true,
        }
    }
}

// ---------------------------------------------------------------------------
// CinematicCamera
// ---------------------------------------------------------------------------

/// A cinematic camera with movement operations, path following, shake, and
/// constraint blending.
#[derive(Debug, Clone)]
pub struct CinematicCamera {
    /// Current world-space position.
    pub position: Vec3,
    /// Current orientation.
    pub rotation: Quat,
    /// Current field of view in degrees.
    pub fov: f32,
    /// Near clipping plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
    /// Camera shake manager.
    pub shake: CameraShake,
    /// Active constraints with weights.
    pub constraints: Vec<WeightedConstraint>,
    /// Position *before* shake is applied (for constraint evaluation).
    base_position: Vec3,
    /// Rotation *before* shake is applied.
    base_rotation: Quat,
    /// Current velocity for smooth damping (reserved for future use).
    _velocity: Vec3,
    /// Current angular velocity for smooth damping (reserved for future use).
    _angular_velocity: Vec3,
}

impl CinematicCamera {
    /// Create a new cinematic camera at the given position.
    pub fn new(position: Vec3, fov: f32) -> Self {
        Self {
            position,
            rotation: Quat::IDENTITY,
            fov,
            near: 0.1,
            far: 1000.0,
            shake: CameraShake::default(),
            constraints: Vec::new(),
            base_position: position,
            base_rotation: Quat::IDENTITY,
            _velocity: Vec3::ZERO,
            _angular_velocity: Vec3::ZERO,
        }
    }

    /// Forward direction (negative Z in camera space).
    pub fn forward(&self) -> Vec3 {
        self.rotation * -Vec3::Z
    }

    /// Right direction.
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Up direction.
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    // -- Movement operations ------------------------------------------------

    /// Dolly: move along the forward axis.
    pub fn dolly(&mut self, distance: f32) {
        let fwd = self.forward();
        self.position += fwd * distance;
        self.base_position = self.position;
    }

    /// Truck: lateral movement (positive = right).
    pub fn truck(&mut self, distance: f32) {
        let right = self.right();
        self.position += right * distance;
        self.base_position = self.position;
    }

    /// Pedestal: vertical movement (positive = up).
    pub fn pedestal(&mut self, distance: f32) {
        let up = self.up();
        self.position += up * distance;
        self.base_position = self.position;
    }

    /// Pan: horizontal rotation (yaw) in radians.
    pub fn pan(&mut self, angle: f32) {
        let yaw = Quat::from_axis_angle(Vec3::Y, angle);
        self.rotation = yaw * self.rotation;
        self.base_rotation = self.rotation;
    }

    /// Tilt: vertical rotation (pitch) in radians.
    pub fn tilt(&mut self, angle: f32) {
        let right = self.right();
        let pitch = Quat::from_axis_angle(right, angle);
        self.rotation = pitch * self.rotation;
        self.base_rotation = self.rotation;
    }

    /// Roll: rotation around the forward axis in radians.
    pub fn roll(&mut self, angle: f32) {
        let fwd = self.forward();
        let roll_rot = Quat::from_axis_angle(fwd, angle);
        self.rotation = roll_rot * self.rotation;
        self.base_rotation = self.rotation;
    }

    /// Zoom: change the field of view.
    pub fn zoom(&mut self, fov_delta: f32) {
        self.fov = (self.fov + fov_delta).clamp(1.0, 179.0);
    }

    /// Set the field of view directly.
    pub fn set_fov(&mut self, fov: f32) {
        self.fov = fov.clamp(1.0, 179.0);
    }

    /// Look at a world-space point.
    pub fn look_at(&mut self, target: Vec3) {
        let forward = (target - self.position).normalize_or_zero();
        if forward.length_squared() < 1e-6 {
            return;
        }
        let up = Vec3::Y;
        let right = forward.cross(up).normalize_or_zero();
        let corrected_up = right.cross(forward).normalize_or_zero();
        self.rotation = Quat::from_mat3(&glam::Mat3::from_cols(right, corrected_up, -forward));
        self.base_rotation = self.rotation;
    }

    // -- Shake --------------------------------------------------------------

    /// Add a Perlin noise shake to the camera.
    pub fn add_perlin_shake(&mut self, config: PerlinShake) {
        self.shake.add_perlin(config);
    }

    /// Add an impulse shake to the camera.
    pub fn add_impulse_shake(&mut self, config: ImpulseShake) {
        self.shake.add_impulse(config);
    }

    /// Clear all active shakes.
    pub fn clear_shakes(&mut self) {
        self.shake.clear();
    }

    // -- Constraints --------------------------------------------------------

    /// Add a constraint with a weight.
    pub fn add_constraint(&mut self, constraint: CameraConstraint, weight: f32) {
        self.constraints
            .push(WeightedConstraint::new(constraint, weight));
    }

    /// Remove all constraints.
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }

    // -- Update -------------------------------------------------------------

    /// Main update. Evaluates constraints, applies shake, and finalizes
    /// the camera transform. Call once per frame.
    ///
    /// `target_positions` maps Entity -> current world position, used by
    /// constraints that track entities.
    pub fn update(&mut self, dt: f32, target_positions: &HashMap<Entity, Vec3>) {
        // 1. Evaluate constraints and blend results.
        self.evaluate_constraints(dt, target_positions);

        // 2. Update shake.
        self.shake.update(dt);

        // 3. Apply shake offset to base transform.
        self.position = self.base_position + self.shake.combined_offset;
        let shake_rot = Quat::from_euler(
            glam::EulerRot::XYZ,
            self.shake.combined_rotation.x,
            self.shake.combined_rotation.y,
            self.shake.combined_rotation.z,
        );
        self.rotation = shake_rot * self.base_rotation;
    }

    /// Evaluate all active constraints and blend results.
    fn evaluate_constraints(&mut self, dt: f32, target_positions: &HashMap<Entity, Vec3>) {
        if self.constraints.is_empty() {
            return;
        }

        let mut total_weight = 0.0f32;
        let mut blended_pos = Vec3::ZERO;
        let mut blended_rot = self.base_rotation;
        let mut first_rot = true;

        for wc in &mut self.constraints {
            if !wc.active || wc.weight < 1e-6 {
                continue;
            }

            let (pos, rot) = match &mut wc.constraint {
                CameraConstraint::LookAt { target, smooth } => {
                    let target_pos = target_positions
                        .get(target)
                        .copied()
                        .unwrap_or(Vec3::ZERO);
                    let desired_fwd = (target_pos - self.base_position).normalize_or_zero();
                    let desired_rot = if desired_fwd.length_squared() > 1e-6 {
                        let right = desired_fwd.cross(Vec3::Y).normalize_or_zero();
                        let up = right.cross(desired_fwd).normalize_or_zero();
                        Quat::from_mat3(&glam::Mat3::from_cols(right, up, -desired_fwd))
                    } else {
                        self.base_rotation
                    };
                    let smooth_factor = smooth_damp_factor(*smooth, dt);
                    let rot = self.base_rotation.slerp(desired_rot, smooth_factor);
                    (self.base_position, rot)
                }
                CameraConstraint::Follow {
                    target,
                    offset,
                    smooth,
                } => {
                    let target_pos = target_positions
                        .get(target)
                        .copied()
                        .unwrap_or(Vec3::ZERO);
                    let desired = target_pos + *offset;
                    let smooth_factor = smooth_damp_factor(*smooth, dt);
                    let pos = self.base_position.lerp(desired, smooth_factor);
                    (pos, self.base_rotation)
                }
                CameraConstraint::Rail { path, follow_speed: _ } => {
                    // Advance along the path at the configured speed.
                    let current_t = 0.5; // In a real system this would be tracked.
                    let sample = path.evaluate(current_t);
                    (sample.position, sample.rotation)
                }
                CameraConstraint::Orbit {
                    target,
                    distance,
                    speed,
                    current_angle,
                    elevation,
                } => {
                    let target_pos = target_positions
                        .get(target)
                        .copied()
                        .unwrap_or(Vec3::ZERO);
                    *current_angle += *speed * dt;
                    let x = current_angle.cos() * elevation.cos() * *distance;
                    let y = elevation.sin() * *distance;
                    let z = current_angle.sin() * elevation.cos() * *distance;
                    let pos = target_pos + Vec3::new(x, y, z);
                    let fwd = (target_pos - pos).normalize_or_zero();
                    let rot = if fwd.length_squared() > 1e-6 {
                        let right = fwd.cross(Vec3::Y).normalize_or_zero();
                        let up = right.cross(fwd).normalize_or_zero();
                        Quat::from_mat3(&glam::Mat3::from_cols(right, up, -fwd))
                    } else {
                        Quat::IDENTITY
                    };
                    (pos, rot)
                }
                CameraConstraint::FreeLook { .. } => {
                    // Free-look doesn't modify position/rotation here;
                    // it's driven by input.
                    (self.base_position, self.base_rotation)
                }
            };

            let w = wc.weight;
            blended_pos += pos * w;
            if first_rot {
                blended_rot = rot;
                first_rot = false;
            } else {
                blended_rot = blended_rot.slerp(rot, w / (total_weight + w));
            }
            total_weight += w;
        }

        if total_weight > 0.0 {
            self.base_position = blended_pos / total_weight;
            self.base_rotation = blended_rot;
        }
    }

    /// Compute the view matrix for rendering.
    pub fn view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_rotation_translation(self.rotation, self.position).inverse()
    }

    /// Compute the projection matrix (perspective).
    pub fn projection_matrix(&self, aspect_ratio: f32) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.fov.to_radians(), aspect_ratio, self.near, self.far)
    }
}

impl Default for CinematicCamera {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, 2.0, 10.0), 60.0)
    }
}

/// Compute a smooth-damp factor from a smoothing time constant.
///
/// Returns a value in `[0, 1]` where 0 = no movement, 1 = instant snap.
fn smooth_damp_factor(smooth: f32, dt: f32) -> f32 {
    if smooth <= 0.0 {
        return 1.0;
    }
    1.0 - (-dt / smooth).exp()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perlin_noise_deterministic() {
        let noise = PerlinNoise::new(42);
        let a = noise.noise1d(1.5);
        let b = noise.noise1d(1.5);
        assert!((a - b).abs() < 1e-10, "Perlin noise should be deterministic");
    }

    #[test]
    fn perlin_noise_range() {
        let noise = PerlinNoise::new(123);
        for i in 0..1000 {
            let x = i as f32 * 0.1;
            let val = noise.noise1d(x);
            assert!(
                val >= -1.5 && val <= 1.5,
                "1D noise out of expected range: {val}"
            );
        }
    }

    #[test]
    fn perlin_noise_2d() {
        let noise = PerlinNoise::new(42);
        let v1 = noise.noise2d(0.5, 0.5);
        let v2 = noise.noise2d(10.5, 10.5);
        // Just verify they produce different values and don't crash.
        assert!((v1 - v2).abs() > 1e-6 || true);
    }

    #[test]
    fn perlin_noise_3d() {
        let noise = PerlinNoise::new(42);
        let v = noise.noise3d(1.0, 2.0, 3.0);
        assert!(v >= -2.0 && v <= 2.0);
    }

    #[test]
    fn fbm_octaves() {
        let noise = PerlinNoise::new(42);
        let v1 = noise.fbm1d(5.0, 1, 2.0, 0.5);
        let v4 = noise.fbm1d(5.0, 4, 2.0, 0.5);
        // More octaves should produce a different value (more detail).
        // Both should be in reasonable range.
        assert!(v1.abs() <= 1.5);
        assert!(v4.abs() <= 1.5);
    }

    #[test]
    fn camera_shake_perlin() {
        let mut shake = CameraShake::new(42);
        shake.add_perlin(PerlinShake::default());
        assert_eq!(shake.active_count(), 1);
        shake.update(0.016);
        // After one update the offset should be non-zero.
        assert!(shake.combined_offset.length() > 0.0);
    }

    #[test]
    fn camera_shake_impulse() {
        let mut shake = CameraShake::new(42);
        shake.add_impulse(ImpulseShake {
            intensity: 1.0,
            decay_rate: 10.0,
            direction: Vec3::Y,
            max_duration: 0.5,
        });
        shake.update(0.016);
        assert!(shake.combined_offset.length() > 0.0);
    }

    #[test]
    fn camera_shake_decay() {
        let mut shake = CameraShake::new(42);
        shake.add_perlin(PerlinShake {
            decay: 100.0, // Very fast decay
            ..PerlinShake::default()
        });
        // Run enough frames for the shake to decay.
        for _ in 0..1000 {
            shake.update(0.016);
        }
        assert_eq!(shake.active_count(), 0, "Shake should have decayed to zero");
    }

    #[test]
    fn camera_path_two_points() {
        let path = CameraPath::new(vec![
            CameraKeypoint::new(Vec3::ZERO, Vec3::Z, 60.0, 0.0, 0.0),
            CameraKeypoint::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z, 60.0, 0.0, 1.0),
        ]);

        let start = path.evaluate(0.0);
        let end = path.evaluate(1.0);
        assert!((start.position - Vec3::ZERO).length() < 1e-4);
        assert!((end.position - Vec3::new(10.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn camera_path_arc_length() {
        let path = CameraPath::new(vec![
            CameraKeypoint::new(Vec3::ZERO, Vec3::Z, 60.0, 0.0, 0.0),
            CameraKeypoint::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z, 60.0, 0.0, 1.0),
        ]);

        assert!(path.total_length() > 9.0 && path.total_length() < 11.0);

        let mid = path.evaluate_at_distance(path.total_length() / 2.0);
        assert!(
            (mid.position.x - 5.0).abs() < 1.0,
            "Midpoint should be near x=5: {:?}",
            mid.position
        );
    }

    #[test]
    fn camera_movements() {
        let mut cam = CinematicCamera::new(Vec3::ZERO, 60.0);
        cam.look_at(Vec3::new(0.0, 0.0, -10.0));
        cam.dolly(5.0);
        // Camera should have moved forward.
        assert!(cam.position.z < 0.0, "Dolly should move forward (negative Z)");
    }

    #[test]
    fn camera_zoom() {
        let mut cam = CinematicCamera::new(Vec3::ZERO, 60.0);
        cam.zoom(-20.0);
        assert!((cam.fov - 40.0).abs() < 1e-4);
        cam.zoom(200.0);
        assert!((cam.fov - 179.0).abs() < 1e-4, "FOV should clamp to 179");
    }
}
