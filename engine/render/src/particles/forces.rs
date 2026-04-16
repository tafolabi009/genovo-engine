// engine/render/src/particles/forces.rs
//
// Force fields that can be applied to particles. Each force implements the
// `ForceField` trait, which takes a particle's position and velocity and
// returns the resulting acceleration (not velocity delta -- the caller
// multiplies by dt).
//
// Includes full implementations of Perlin/simplex noise for turbulence
// and curl noise for divergence-free fluid-like motion.

use glam::Vec3;

// ---------------------------------------------------------------------------
// ForceField trait
// ---------------------------------------------------------------------------

/// A force that can be applied to particles.
///
/// Implementations must be `Send + Sync` so that force fields can be shared
/// across threads when the particle system is updated in parallel.
pub trait ForceField: Send + Sync {
    /// Computes the acceleration to apply to a particle.
    ///
    /// # Arguments
    /// * `pos` - The particle's current world-space position.
    /// * `vel` - The particle's current velocity.
    /// * `dt`  - The frame delta time (some forces need it for integration).
    /// * `time` - The global elapsed time (for animated forces).
    ///
    /// # Returns
    /// The acceleration vector (force / mass, assuming unit mass).
    fn acceleration(&self, pos: Vec3, vel: Vec3, dt: f32, time: f32) -> Vec3;

    /// Returns a human-readable name for debugging.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// GravityForce
// ---------------------------------------------------------------------------

/// A constant directional force (like gravity).
#[derive(Debug, Clone)]
pub struct GravityForce {
    /// Direction of the force (typically `Vec3::NEG_Y`).
    pub direction: Vec3,
    /// Strength in units per second squared.
    pub strength: f32,
}

impl GravityForce {
    /// Earth-like gravity pointing down.
    pub fn earth() -> Self {
        Self {
            direction: Vec3::NEG_Y,
            strength: 9.81,
        }
    }

    /// Custom gravity.
    pub fn new(direction: Vec3, strength: f32) -> Self {
        Self {
            direction: direction.normalize(),
            strength,
        }
    }
}

impl ForceField for GravityForce {
    fn acceleration(&self, _pos: Vec3, _vel: Vec3, _dt: f32, _time: f32) -> Vec3 {
        self.direction * self.strength
    }

    fn name(&self) -> &'static str {
        "GravityForce"
    }
}

// ---------------------------------------------------------------------------
// WindForce
// ---------------------------------------------------------------------------

/// A directional wind force with optional turbulence.
#[derive(Debug, Clone)]
pub struct WindForce {
    /// Base wind direction (will be normalized).
    pub direction: Vec3,
    /// Base wind strength.
    pub strength: f32,
    /// Turbulence amplitude (random variation in strength).
    pub turbulence: f32,
    /// Turbulence frequency (how fast the turbulence changes).
    pub frequency: f32,
}

impl WindForce {
    pub fn new(direction: Vec3, strength: f32) -> Self {
        Self {
            direction: direction.normalize(),
            strength,
            turbulence: 0.0,
            frequency: 1.0,
        }
    }

    pub fn with_turbulence(mut self, turbulence: f32, frequency: f32) -> Self {
        self.turbulence = turbulence;
        self.frequency = frequency;
        self
    }
}

impl ForceField for WindForce {
    fn acceleration(&self, pos: Vec3, _vel: Vec3, _dt: f32, time: f32) -> Vec3 {
        let base = self.direction * self.strength;
        if self.turbulence > 0.0 {
            // Use position and time to create a spatially-varying turbulence.
            let noise_input = pos * self.frequency + Vec3::splat(time * self.frequency);
            let nx = value_noise_3d(noise_input.x, noise_input.y, noise_input.z);
            let ny = value_noise_3d(
                noise_input.x + 31.416,
                noise_input.y + 47.853,
                noise_input.z + 12.345,
            );
            let nz = value_noise_3d(
                noise_input.x + 73.156,
                noise_input.y + 19.247,
                noise_input.z + 58.912,
            );
            let turbulence_vec = Vec3::new(nx, ny, nz) * self.turbulence;
            base + turbulence_vec
        } else {
            base
        }
    }

    fn name(&self) -> &'static str {
        "WindForce"
    }
}

// ---------------------------------------------------------------------------
// VortexForce
// ---------------------------------------------------------------------------

/// A vortex (tornado/whirlpool) force that spirals particles around an axis.
#[derive(Debug, Clone)]
pub struct VortexForce {
    /// Center of the vortex (world space).
    pub center: Vec3,
    /// Axis of rotation (unit vector).
    pub axis: Vec3,
    /// Tangential strength (rotation speed).
    pub strength: f32,
    /// Inward pull towards the axis. Positive = attract, negative = repel.
    pub inward_pull: f32,
    /// Upward pull along the axis.
    pub upward_pull: f32,
}

impl VortexForce {
    pub fn new(center: Vec3, axis: Vec3, strength: f32) -> Self {
        Self {
            center,
            axis: axis.normalize(),
            strength,
            inward_pull: 0.0,
            upward_pull: 0.0,
        }
    }

    pub fn with_inward_pull(mut self, pull: f32) -> Self {
        self.inward_pull = pull;
        self
    }

    pub fn with_upward_pull(mut self, pull: f32) -> Self {
        self.upward_pull = pull;
        self
    }
}

impl ForceField for VortexForce {
    fn acceleration(&self, pos: Vec3, _vel: Vec3, _dt: f32, _time: f32) -> Vec3 {
        // Project the particle position onto the axis plane.
        let to_particle = pos - self.center;
        let along_axis = self.axis * to_particle.dot(self.axis);
        let radial = to_particle - along_axis;
        let dist = radial.length();

        if dist < 1e-6 {
            return self.axis * self.upward_pull;
        }

        let radial_dir = radial / dist;
        // Tangential direction: cross(axis, radial_dir)
        let tangent = self.axis.cross(radial_dir);

        // Tangential acceleration (creates rotation).
        let tangential_acc = tangent * self.strength;
        // Inward acceleration (pulls towards axis).
        let inward_acc = -radial_dir * self.inward_pull;
        // Upward acceleration (along axis).
        let upward_acc = self.axis * self.upward_pull;

        tangential_acc + inward_acc + upward_acc
    }

    fn name(&self) -> &'static str {
        "VortexForce"
    }
}

// ---------------------------------------------------------------------------
// AttractorForce
// ---------------------------------------------------------------------------

/// A point attractor or repulsor.
#[derive(Debug, Clone)]
pub struct AttractorForce {
    /// Position of the attractor (world space).
    pub position: Vec3,
    /// Strength. Positive = attract, negative = repel.
    pub strength: f32,
    /// Radius of effect. Particles beyond this distance are unaffected.
    pub radius: f32,
    /// Falloff exponent. 0 = constant, 1 = linear, 2 = inverse-square.
    pub falloff: f32,
    /// Dead zone radius. Particles closer than this are unaffected (prevents
    /// infinite acceleration at zero distance).
    pub dead_zone: f32,
}

impl AttractorForce {
    pub fn new(position: Vec3, strength: f32, radius: f32) -> Self {
        Self {
            position,
            strength,
            radius,
            falloff: 2.0,
            dead_zone: 0.1,
        }
    }

    pub fn with_falloff(mut self, falloff: f32) -> Self {
        self.falloff = falloff;
        self
    }

    pub fn with_dead_zone(mut self, dead_zone: f32) -> Self {
        self.dead_zone = dead_zone;
        self
    }
}

impl ForceField for AttractorForce {
    fn acceleration(&self, pos: Vec3, _vel: Vec3, _dt: f32, _time: f32) -> Vec3 {
        let to_attractor = self.position - pos;
        let dist = to_attractor.length();

        if dist < self.dead_zone || dist > self.radius {
            return Vec3::ZERO;
        }

        let dir = to_attractor / dist;

        // Compute falloff.
        let normalized_dist = dist / self.radius;
        let attenuation = match self.falloff as u32 {
            0 => 1.0,
            1 => 1.0 - normalized_dist,
            2 => {
                let inv = 1.0 / (dist * dist).max(0.01);
                inv.min(100.0) // Clamp to prevent explosion.
            }
            _ => (1.0 - normalized_dist).powf(self.falloff),
        };

        dir * self.strength * attenuation
    }

    fn name(&self) -> &'static str {
        "AttractorForce"
    }
}

// ---------------------------------------------------------------------------
// DragForce
// ---------------------------------------------------------------------------

/// Velocity-proportional drag (air resistance / friction).
///
/// The acceleration is `-coefficient * velocity`, which exponentially
/// decays velocity over time.
#[derive(Debug, Clone)]
pub struct DragForce {
    /// Drag coefficient. Higher = more drag.
    pub coefficient: f32,
}

impl DragForce {
    pub fn new(coefficient: f32) -> Self {
        Self { coefficient }
    }
}

impl ForceField for DragForce {
    fn acceleration(&self, _pos: Vec3, vel: Vec3, _dt: f32, _time: f32) -> Vec3 {
        -vel * self.coefficient
    }

    fn name(&self) -> &'static str {
        "DragForce"
    }
}

// ---------------------------------------------------------------------------
// TurbulenceForce
// ---------------------------------------------------------------------------

/// 3D noise-based turbulence force.
///
/// Uses fractal Brownian motion (fBm) with multiple octaves of value noise
/// to create organic, chaotic motion.
#[derive(Debug, Clone)]
pub struct TurbulenceForce {
    /// Overall turbulence strength.
    pub strength: f32,
    /// Base frequency of the noise.
    pub frequency: f32,
    /// Number of noise octaves.
    pub octaves: u32,
    /// Lacunarity: frequency multiplier per octave (typically 2.0).
    pub lacunarity: f32,
    /// Persistence: amplitude multiplier per octave (typically 0.5).
    pub persistence: f32,
    /// Scroll speed: how fast the noise field moves through time.
    pub scroll_speed: f32,
}

impl TurbulenceForce {
    pub fn new(strength: f32, frequency: f32) -> Self {
        Self {
            strength,
            frequency,
            octaves: 3,
            lacunarity: 2.0,
            persistence: 0.5,
            scroll_speed: 1.0,
        }
    }

    pub fn with_octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    pub fn with_lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }

    pub fn with_persistence(mut self, persistence: f32) -> Self {
        self.persistence = persistence;
        self
    }
}

impl ForceField for TurbulenceForce {
    fn acceleration(&self, pos: Vec3, _vel: Vec3, _dt: f32, time: f32) -> Vec3 {
        let t = time * self.scroll_speed;

        // Sample noise at 3 offset positions to get a 3D vector field.
        let p = pos * self.frequency + Vec3::splat(t);

        let nx = fbm_3d(
            p.x, p.y, p.z,
            self.octaves, self.lacunarity, self.persistence,
        );
        let ny = fbm_3d(
            p.x + 31.416, p.y + 47.853, p.z + 12.345,
            self.octaves, self.lacunarity, self.persistence,
        );
        let nz = fbm_3d(
            p.x + 73.156, p.y + 19.247, p.z + 58.912,
            self.octaves, self.lacunarity, self.persistence,
        );

        Vec3::new(nx, ny, nz) * self.strength
    }

    fn name(&self) -> &'static str {
        "TurbulenceForce"
    }
}

// ---------------------------------------------------------------------------
// NoiseField
// ---------------------------------------------------------------------------

/// A 3D value noise field with fBm, usable as a standalone module.
///
/// Not a force field itself, but provides noise sampling that other systems
/// (volumetric fog, terrain, etc.) can use.
#[derive(Debug, Clone)]
pub struct NoiseField {
    /// Base frequency.
    pub frequency: f32,
    /// Number of octaves.
    pub octaves: u32,
    /// Lacunarity (frequency multiplier per octave).
    pub lacunarity: f32,
    /// Persistence (amplitude multiplier per octave).
    pub persistence: f32,
    /// Offset applied to the input coordinates.
    pub offset: Vec3,
}

impl NoiseField {
    pub fn new(frequency: f32, octaves: u32) -> Self {
        Self {
            frequency,
            octaves,
            lacunarity: 2.0,
            persistence: 0.5,
            offset: Vec3::ZERO,
        }
    }

    /// Samples the noise field at the given position.
    /// Returns a value in approximately [-1, 1].
    pub fn sample(&self, pos: Vec3) -> f32 {
        let p = (pos + self.offset) * self.frequency;
        fbm_3d(p.x, p.y, p.z, self.octaves, self.lacunarity, self.persistence)
    }

    /// Samples the noise field at a given position and time.
    pub fn sample_4d(&self, pos: Vec3, time: f32) -> f32 {
        let p = (pos + self.offset) * self.frequency;
        fbm_3d(
            p.x + time * 0.3,
            p.y + time * 0.2,
            p.z + time * 0.1,
            self.octaves,
            self.lacunarity,
            self.persistence,
        )
    }

    /// Computes the gradient of the noise field at `pos` using central
    /// differences (for normal computation, etc.).
    pub fn gradient(&self, pos: Vec3, epsilon: f32) -> Vec3 {
        let dx = self.sample(pos + Vec3::X * epsilon)
            - self.sample(pos - Vec3::X * epsilon);
        let dy = self.sample(pos + Vec3::Y * epsilon)
            - self.sample(pos - Vec3::Y * epsilon);
        let dz = self.sample(pos + Vec3::Z * epsilon)
            - self.sample(pos - Vec3::Z * epsilon);
        Vec3::new(dx, dy, dz) / (2.0 * epsilon)
    }
}

impl Default for NoiseField {
    fn default() -> Self {
        Self::new(1.0, 4)
    }
}

// ---------------------------------------------------------------------------
// CurlNoise
// ---------------------------------------------------------------------------

/// Curl noise produces a divergence-free vector field, perfect for
/// fluid-like, smoke-like particle motion. Particles following curl noise
/// naturally form swirls and eddies without converging to a point.
///
/// The curl is computed as the cross product of the gradient of three
/// independent scalar noise fields.
#[derive(Debug, Clone)]
pub struct CurlNoise {
    /// Base frequency.
    pub frequency: f32,
    /// Number of noise octaves.
    pub octaves: u32,
    /// Lacunarity.
    pub lacunarity: f32,
    /// Persistence.
    pub persistence: f32,
    /// Strength multiplier.
    pub strength: f32,
    /// Epsilon for finite-difference gradient computation.
    pub epsilon: f32,
    /// Time scroll speed.
    pub scroll_speed: f32,
}

impl CurlNoise {
    pub fn new(strength: f32, frequency: f32) -> Self {
        Self {
            frequency,
            octaves: 3,
            lacunarity: 2.0,
            persistence: 0.5,
            strength,
            epsilon: 0.01,
            scroll_speed: 1.0,
        }
    }

    pub fn with_octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    /// Samples the curl noise field at the given position and time.
    ///
    /// Returns a divergence-free 3D vector.
    pub fn sample(&self, pos: Vec3, time: f32) -> Vec3 {
        let t = time * self.scroll_speed;
        let p = pos * self.frequency;
        let eps = self.epsilon;

        // We need two independent scalar fields. The curl of a scalar potential
        // in 3D is defined via the cross product of the gradient.
        // For a 3D divergence-free field, we use:
        //   curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
        // where Fx, Fy, Fz are three independent noise fields.

        // Sample 3 independent noise fields at offset positions.
        let sample_noise = |ox: f32, oy: f32, oz: f32| -> f32 {
            fbm_3d(
                p.x + ox + t * 0.13,
                p.y + oy + t * 0.07,
                p.z + oz + t * 0.11,
                self.octaves,
                self.lacunarity,
                self.persistence,
            )
        };

        // Field 1 offsets.
        let f1_ox = 0.0;
        let f1_oy = 0.0;
        let f1_oz = 0.0;

        // Field 2 offsets.
        let f2_ox = 31.416;
        let f2_oy = 47.853;
        let f2_oz = 12.345;

        // Field 3 offsets.
        let f3_ox = 73.156;
        let f3_oy = 19.247;
        let f3_oz = 58.912;

        // dF1/dy, dF1/dz via central differences.
        let df1_dy = (sample_noise(f1_ox, f1_oy + eps, f1_oz)
            - sample_noise(f1_ox, f1_oy - eps, f1_oz))
            / (2.0 * eps);
        let df1_dz = (sample_noise(f1_ox, f1_oy, f1_oz + eps)
            - sample_noise(f1_ox, f1_oy, f1_oz - eps))
            / (2.0 * eps);

        // dF2/dx, dF2/dz
        let df2_dx = (sample_noise(f2_ox + eps, f2_oy, f2_oz)
            - sample_noise(f2_ox - eps, f2_oy, f2_oz))
            / (2.0 * eps);
        let df2_dz = (sample_noise(f2_ox, f2_oy, f2_oz + eps)
            - sample_noise(f2_ox, f2_oy, f2_oz - eps))
            / (2.0 * eps);

        // dF3/dx, dF3/dy
        let df3_dx = (sample_noise(f3_ox + eps, f3_oy, f3_oz)
            - sample_noise(f3_ox - eps, f3_oy, f3_oz))
            / (2.0 * eps);
        let df3_dy = (sample_noise(f3_ox, f3_oy + eps, f3_oz)
            - sample_noise(f3_ox, f3_oy - eps, f3_oz))
            / (2.0 * eps);

        // curl = (dF3/dy - dF2/dz, dF1/dz - dF3/dx, dF2/dx - dF1/dy)
        Vec3::new(
            df3_dy - df2_dz,
            df1_dz - df3_dx,
            df2_dx - df1_dy,
        ) * self.strength
    }
}

impl ForceField for CurlNoise {
    fn acceleration(&self, pos: Vec3, _vel: Vec3, _dt: f32, time: f32) -> Vec3 {
        self.sample(pos, time)
    }

    fn name(&self) -> &'static str {
        "CurlNoise"
    }
}

// ===========================================================================
// Noise functions (implemented from scratch)
// ===========================================================================

/// Permutation table for hash-based noise. This is a standard shuffled
/// table of 0..255, doubled for wrap-around indexing.
const PERM: [u8; 512] = {
    let base: [u8; 256] = [
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
        140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
        247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
        57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
        74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
        60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
        65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
        200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
        52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
        207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
        119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
        129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
        218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
        81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
        184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
        222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
    ];
    let mut table = [0u8; 512];
    let mut i = 0;
    while i < 256 {
        table[i] = base[i];
        table[i + 256] = base[i];
        i += 1;
    }
    table
};

/// Hash function for noise: maps integer coordinates to a pseudo-random value.
#[inline]
fn hash(x: i32, y: i32, z: i32) -> u8 {
    let x = (x & 255) as usize;
    let y = (y & 255) as usize;
    let z = (z & 255) as usize;
    PERM[PERM[PERM[x] as usize + y] as usize + z]
}

/// Quintic smoothstep (Perlin's improved noise uses this instead of
/// Hermite smoothstep for C2 continuity).
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// 3D value noise.
///
/// Returns a value in approximately [-1, 1]. Uses trilinear interpolation
/// of hashed lattice values with quintic fade curves.
pub fn value_noise_3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;

    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    // Hash the 8 corners of the unit cube.
    let c000 = hash(xi, yi, zi) as f32 / 128.0 - 1.0;
    let c100 = hash(xi + 1, yi, zi) as f32 / 128.0 - 1.0;
    let c010 = hash(xi, yi + 1, zi) as f32 / 128.0 - 1.0;
    let c110 = hash(xi + 1, yi + 1, zi) as f32 / 128.0 - 1.0;
    let c001 = hash(xi, yi, zi + 1) as f32 / 128.0 - 1.0;
    let c101 = hash(xi + 1, yi, zi + 1) as f32 / 128.0 - 1.0;
    let c011 = hash(xi, yi + 1, zi + 1) as f32 / 128.0 - 1.0;
    let c111 = hash(xi + 1, yi + 1, zi + 1) as f32 / 128.0 - 1.0;

    // Trilinear interpolation.
    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);

    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);

    lerp(y0, y1, w)
}

/// Gradient vectors for 3D Perlin noise (12 edges of a cube).
const GRAD3: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
];

/// Gradient dot product for Perlin noise.
#[inline]
fn grad_dot(hash_val: u8, x: f32, y: f32, z: f32) -> f32 {
    let g = &GRAD3[(hash_val % 12) as usize];
    g[0] * x + g[1] * y + g[2] * z
}

/// 3D Perlin noise (improved, with gradient-based interpolation).
///
/// Returns a value in approximately [-1, 1].
pub fn perlin_noise_3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;

    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    // Hash the 8 corners.
    let h000 = hash(xi, yi, zi);
    let h100 = hash(xi + 1, yi, zi);
    let h010 = hash(xi, yi + 1, zi);
    let h110 = hash(xi + 1, yi + 1, zi);
    let h001 = hash(xi, yi, zi + 1);
    let h101 = hash(xi + 1, yi, zi + 1);
    let h011 = hash(xi, yi + 1, zi + 1);
    let h111 = hash(xi + 1, yi + 1, zi + 1);

    // Gradient dot products.
    let g000 = grad_dot(h000, xf, yf, zf);
    let g100 = grad_dot(h100, xf - 1.0, yf, zf);
    let g010 = grad_dot(h010, xf, yf - 1.0, zf);
    let g110 = grad_dot(h110, xf - 1.0, yf - 1.0, zf);
    let g001 = grad_dot(h001, xf, yf, zf - 1.0);
    let g101 = grad_dot(h101, xf - 1.0, yf, zf - 1.0);
    let g011 = grad_dot(h011, xf, yf - 1.0, zf - 1.0);
    let g111 = grad_dot(h111, xf - 1.0, yf - 1.0, zf - 1.0);

    // Trilinear interpolation.
    let x00 = lerp(g000, g100, u);
    let x10 = lerp(g010, g110, u);
    let x01 = lerp(g001, g101, u);
    let x11 = lerp(g011, g111, u);

    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);

    lerp(y0, y1, w)
}

/// Fractal Brownian Motion: sums multiple octaves of noise with
/// decreasing amplitude and increasing frequency.
pub fn fbm_3d(
    x: f32, y: f32, z: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        value += perlin_noise_3d(x * frequency, y * frequency, z * frequency) * amplitude;
        max_amplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    // Normalize to [-1, 1].
    if max_amplitude > 0.0 {
        value / max_amplitude
    } else {
        0.0
    }
}

/// Simple linear interpolation.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Simplex Noise (3D)
// ---------------------------------------------------------------------------

/// 3D simplex noise implementation.
///
/// Simplex noise has several advantages over Perlin noise:
/// - Fewer multiplications (O(n) vs O(2^n) corners).
/// - No visible axis-aligned artifacts.
/// - Better gradient distribution.
pub fn simplex_noise_3d(x: f32, y: f32, z: f32) -> f32 {
    // Skewing and unskewing factors for 3D.
    const F3: f32 = 1.0 / 3.0;
    const G3: f32 = 1.0 / 6.0;

    // Skew the input space to determine which simplex cell we're in.
    let s = (x + y + z) * F3;
    let i = (x + s).floor() as i32;
    let j = (y + s).floor() as i32;
    let k = (z + s).floor() as i32;

    let t = (i + j + k) as f32 * G3;
    // Unskewed cell origin.
    let x0 = x - (i as f32 - t);
    let y0 = y - (j as f32 - t);
    let z0 = z - (k as f32 - t);

    // Determine which simplex we are in.
    let (i1, j1, k1, i2, j2, k2);
    if x0 >= y0 {
        if y0 >= z0 {
            // X Y Z order.
            i1 = 1; j1 = 0; k1 = 0;
            i2 = 1; j2 = 1; k2 = 0;
        } else if x0 >= z0 {
            // X Z Y order.
            i1 = 1; j1 = 0; k1 = 0;
            i2 = 1; j2 = 0; k2 = 1;
        } else {
            // Z X Y order.
            i1 = 0; j1 = 0; k1 = 1;
            i2 = 1; j2 = 0; k2 = 1;
        }
    } else {
        if y0 < z0 {
            // Z Y X order.
            i1 = 0; j1 = 0; k1 = 1;
            i2 = 0; j2 = 1; k2 = 1;
        } else if x0 < z0 {
            // Y Z X order.
            i1 = 0; j1 = 1; k1 = 0;
            i2 = 0; j2 = 1; k2 = 1;
        } else {
            // Y X Z order.
            i1 = 0; j1 = 1; k1 = 0;
            i2 = 1; j2 = 1; k2 = 0;
        }
    }

    // Offsets for second corner.
    let x1 = x0 - i1 as f32 + G3;
    let y1 = y0 - j1 as f32 + G3;
    let z1 = z0 - k1 as f32 + G3;

    // Offsets for third corner.
    let x2 = x0 - i2 as f32 + 2.0 * G3;
    let y2 = y0 - j2 as f32 + 2.0 * G3;
    let z2 = z0 - k2 as f32 + 2.0 * G3;

    // Offsets for fourth corner.
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;

    // Calculate contribution from each corner.
    let mut n = 0.0;

    let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if t0 >= 0.0 {
        let t0 = t0 * t0;
        n += t0 * t0 * grad_dot(hash(i, j, k), x0, y0, z0);
    }

    let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if t1 >= 0.0 {
        let t1 = t1 * t1;
        n += t1 * t1 * grad_dot(hash(i + i1, j + j1, k + k1), x1, y1, z1);
    }

    let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if t2 >= 0.0 {
        let t2 = t2 * t2;
        n += t2 * t2 * grad_dot(hash(i + i2, j + j2, k + k2), x2, y2, z2);
    }

    let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if t3 >= 0.0 {
        let t3 = t3 * t3;
        n += t3 * t3 * grad_dot(hash(i + 1, j + 1, k + 1), x3, y3, z3);
    }

    // Scale to [-1, 1].
    32.0 * n
}

/// Fractal Brownian Motion using simplex noise.
pub fn fbm_simplex_3d(
    x: f32, y: f32, z: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        value += simplex_noise_3d(
            x * frequency,
            y * frequency,
            z * frequency,
        ) * amplitude;
        max_amplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    if max_amplitude > 0.0 {
        value / max_amplitude
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// ForceFieldSet -- collection of forces
// ---------------------------------------------------------------------------

/// A collection of force fields that can be applied together.
pub struct ForceFieldSet {
    forces: Vec<Box<dyn ForceField>>,
}

impl ForceFieldSet {
    pub fn new() -> Self {
        Self {
            forces: Vec::new(),
        }
    }

    /// Adds a force to the set.
    pub fn add<F: ForceField + 'static>(&mut self, force: F) {
        self.forces.push(Box::new(force));
    }

    /// Removes all forces.
    pub fn clear(&mut self) {
        self.forces.clear();
    }

    /// Returns the number of active forces.
    pub fn len(&self) -> usize {
        self.forces.len()
    }

    /// Returns `true` if there are no forces.
    pub fn is_empty(&self) -> bool {
        self.forces.is_empty()
    }

    /// Computes the combined acceleration from all forces at a given position.
    pub fn combined_acceleration(
        &self,
        pos: Vec3,
        vel: Vec3,
        dt: f32,
        time: f32,
    ) -> Vec3 {
        let mut acc = Vec3::ZERO;
        for force in &self.forces {
            acc += force.acceleration(pos, vel, dt, time);
        }
        acc
    }

    /// Applies all forces to a particle pool.
    pub fn apply_to_pool(
        &self,
        pool: &mut super::particle::ParticlePool,
        dt: f32,
        time: f32,
    ) {
        if self.forces.is_empty() {
            return;
        }
        for i in 0..pool.alive_count {
            let pos = pool.positions[i];
            let vel = pool.velocities[i];
            let acc = self.combined_acceleration(pos, vel, dt, time);
            pool.velocities[i] += acc * dt;
        }
    }
}

impl Default for ForceFieldSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_force_applies_downward() {
        let g = GravityForce::earth();
        let acc = g.acceleration(Vec3::ZERO, Vec3::ZERO, 0.016, 0.0);
        assert!(acc.y < 0.0);
        assert!((acc.y + 9.81).abs() < 0.01);
    }

    #[test]
    fn drag_slows_particles() {
        let drag = DragForce::new(5.0);
        let vel = Vec3::new(10.0, 0.0, 0.0);
        let acc = drag.acceleration(Vec3::ZERO, vel, 0.016, 0.0);
        // Acceleration should oppose velocity.
        assert!(acc.x < 0.0);
    }

    #[test]
    fn value_noise_is_bounded() {
        for i in 0..1000 {
            let x = i as f32 * 0.1;
            let y = i as f32 * 0.073;
            let z = i as f32 * 0.051;
            let v = value_noise_3d(x, y, z);
            assert!(v >= -1.1 && v <= 1.1, "Noise out of range: {v}");
        }
    }

    #[test]
    fn perlin_noise_is_bounded() {
        for i in 0..1000 {
            let x = i as f32 * 0.1;
            let y = i as f32 * 0.073;
            let z = i as f32 * 0.051;
            let v = perlin_noise_3d(x, y, z);
            assert!(v >= -1.5 && v <= 1.5, "Perlin noise out of range: {v}");
        }
    }

    #[test]
    fn simplex_noise_is_bounded() {
        for i in 0..1000 {
            let x = i as f32 * 0.1;
            let y = i as f32 * 0.073;
            let z = i as f32 * 0.051;
            let v = simplex_noise_3d(x, y, z);
            assert!(v >= -2.0 && v <= 2.0, "Simplex noise out of range: {v}");
        }
    }

    #[test]
    fn vortex_creates_tangential_force() {
        let vortex = VortexForce::new(Vec3::ZERO, Vec3::Y, 10.0);
        let pos = Vec3::new(1.0, 0.0, 0.0);
        let acc = vortex.acceleration(pos, Vec3::ZERO, 0.016, 0.0);
        // Tangential force should be primarily in the Z direction for a
        // particle on the X axis with Y-axis rotation.
        assert!(acc.z.abs() > 0.1);
    }

    #[test]
    fn attractor_pulls_towards_center() {
        let attr = AttractorForce::new(Vec3::ZERO, 10.0, 100.0);
        let pos = Vec3::new(5.0, 0.0, 0.0);
        let acc = attr.acceleration(pos, Vec3::ZERO, 0.016, 0.0);
        // Should pull towards origin (negative X).
        assert!(acc.x < 0.0);
    }

    #[test]
    fn curl_noise_is_divergence_free_approximately() {
        // The divergence of a curl field should be approximately zero.
        let curl = CurlNoise::new(1.0, 1.0);
        let eps = 0.001;
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let time = 0.5;

        let dvx_dx = (curl.sample(pos + Vec3::X * eps, time).x
            - curl.sample(pos - Vec3::X * eps, time).x)
            / (2.0 * eps);
        let dvy_dy = (curl.sample(pos + Vec3::Y * eps, time).y
            - curl.sample(pos - Vec3::Y * eps, time).y)
            / (2.0 * eps);
        let dvz_dz = (curl.sample(pos + Vec3::Z * eps, time).z
            - curl.sample(pos - Vec3::Z * eps, time).z)
            / (2.0 * eps);

        let div = dvx_dx + dvy_dy + dvz_dz;
        assert!(
            div.abs() < 1.0,
            "Divergence should be approximately zero, got {div}"
        );
    }

    #[test]
    fn force_field_set_combines() {
        let mut set = ForceFieldSet::new();
        set.add(GravityForce::earth());
        set.add(WindForce::new(Vec3::X, 5.0));

        let acc = set.combined_acceleration(Vec3::ZERO, Vec3::ZERO, 0.016, 0.0);
        assert!(acc.y < 0.0, "Should have downward gravity");
        assert!(acc.x > 0.0, "Should have rightward wind");
    }
}
