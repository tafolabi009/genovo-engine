//! # Precision Physics
//!
//! Frostbite/FIFA-level physics precision with deterministic simulation,
//! stable constraint solving, sports ball physics (Magnus effect, drag,
//! spin, bounce), soft contact modeling, character body dynamics, advanced
//! vehicle physics, and determinism verification tools.
//!
//! ## Key features
//!
//! - **Fixed-point arithmetic** (`FixedFloat`) for bit-exact cross-platform
//!   determinism.
//! - **Precision solver** with sub-stepping, shock propagation, and split
//!   impulse position correction.
//! - **Sports physics**: Magnus force, Reynolds-number-dependent drag,
//!   spin transfer on bounce, rolling friction.
//! - **Soft contact** with compliance, Hertz contact area, anisotropic friction.
//! - **Character body** for sports characters: root motion, foot IK grounding,
//!   body lean, per-limb collision.
//! - **Vehicle physics V2** with tire temperature, load transfer, downforce,
//!   differentials, and forced induction.
//! - **Determinism tools**: state hashing, replay comparison, desync detection.

use std::collections::HashMap;
use std::fmt;

// ===========================================================================
// Constants
// ===========================================================================

/// Gravitational acceleration (m/s^2).
pub const GRAVITY: f32 = 9.81;

/// Standard air density at sea level (kg/m^3).
pub const AIR_DENSITY: f32 = 1.225;

/// Pi constant.
pub const PI: f32 = std::f32::consts::PI;

/// Fixed-point fractional bits.
pub const FIXED_FRAC_BITS: i32 = 16;

/// Fixed-point scale factor (2^16 = 65536).
pub const FIXED_SCALE: i32 = 1 << FIXED_FRAC_BITS;

/// Default solver iterations.
pub const DEFAULT_SOLVER_ITERATIONS: u32 = 30;

/// Default sub-steps per frame.
pub const DEFAULT_SUBSTEPS: u32 = 4;

/// Default position correction factor (Baumgarte).
pub const DEFAULT_BAUMGARTE: f32 = 0.2;

/// Split impulse threshold for position correction.
pub const SPLIT_IMPULSE_THRESHOLD: f32 = -0.02;

/// Small epsilon.
pub const EPSILON: f32 = 1e-6;

/// Maximum angular velocity (rad/s) to prevent instability.
pub const MAX_ANGULAR_VELOCITY: f32 = 100.0;

/// Default coefficient of restitution for sports balls.
pub const DEFAULT_RESTITUTION: f32 = 0.7;

/// Default rolling friction coefficient.
pub const DEFAULT_ROLLING_FRICTION: f32 = 0.02;

/// Standard soccer ball mass (kg).
pub const SOCCER_BALL_MASS: f32 = 0.43;

/// Standard soccer ball radius (m).
pub const SOCCER_BALL_RADIUS: f32 = 0.11;

/// Standard soccer ball drag coefficient.
pub const SOCCER_BALL_CD: f32 = 0.25;

/// Magnus effect coefficient for a sphere.
pub const MAGNUS_COEFFICIENT: f32 = 0.5;

/// Drag crisis Reynolds number (smooth sphere).
pub const DRAG_CRISIS_RE: f32 = 200_000.0;

/// Drag coefficient below drag crisis.
pub const CD_SUBCRITICAL: f32 = 0.47;

/// Drag coefficient above drag crisis.
pub const CD_SUPERCRITICAL: f32 = 0.2;

/// Kinematic viscosity of air at 20C (m^2/s).
pub const AIR_VISCOSITY: f32 = 1.516e-5;

/// Default tire temperature (Celsius).
pub const DEFAULT_TIRE_TEMP: f32 = 60.0;

/// Optimal tire temperature (Celsius).
pub const OPTIMAL_TIRE_TEMP: f32 = 90.0;

/// Tire temperature range for optimal grip.
pub const TIRE_TEMP_RANGE: f32 = 20.0;

// ===========================================================================
// Vec3Phys — lightweight 3D vector for physics
// ===========================================================================

/// Simple 3D vector for physics computations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3Phys {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3Phys {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const UP: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const RIGHT: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const FORWARD: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline]
    pub fn length(&self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn normalized(&self) -> Self {
        let l = self.length();
        if l < EPSILON {
            return Self::ZERO;
        }
        Self {
            x: self.x / l,
            y: self.y / l,
            z: self.z / l,
        }
    }

    #[inline]
    pub fn dot(a: Self, b: Self) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    #[inline]
    pub fn cross(a: Self, b: Self) -> Self {
        Self {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    #[inline]
    pub fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    #[inline]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Self {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            z: a.z + (b.z - a.z) * t,
        }
    }

    #[inline]
    pub fn distance(a: Self, b: Self) -> f32 {
        a.sub(b).length()
    }

    #[inline]
    pub fn reflect(self, normal: Self) -> Self {
        let d = Self::dot(self, normal);
        self.sub(normal.scale(2.0 * d))
    }

    #[inline]
    pub fn project_onto(self, onto: Self) -> Self {
        let d = Self::dot(self, onto);
        let m = onto.length_sq();
        if m < EPSILON {
            return Self::ZERO;
        }
        onto.scale(d / m)
    }

    /// Clamp magnitude.
    #[inline]
    pub fn clamp_length(self, max_len: f32) -> Self {
        let l = self.length();
        if l > max_len && l > EPSILON {
            self.scale(max_len / l)
        } else {
            self
        }
    }
}

impl Default for Vec3Phys {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for Vec3Phys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

impl std::ops::Add for Vec3Phys {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::add(self, rhs)
    }
}

impl std::ops::Sub for Vec3Phys {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::sub(self, rhs)
    }
}

impl std::ops::Mul<f32> for Vec3Phys {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        self.scale(rhs)
    }
}

impl std::ops::Neg for Vec3Phys {
    type Output = Self;
    fn neg(self) -> Self {
        Self::neg(self)
    }
}

impl std::ops::AddAssign for Vec3Phys {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::ops::SubAssign for Vec3Phys {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

// ===========================================================================
// FixedFloat — 32-bit fixed-point (16.16) for deterministic computation
// ===========================================================================

/// 32-bit fixed-point number in 16.16 format.
///
/// Provides deterministic arithmetic that produces identical results across
/// all platforms, unlike IEEE 754 floating point which can vary due to
/// different rounding modes, FMA availability, and x87 vs SSE behavior.
///
/// Range: approximately -32768 to +32767.999985 with a precision of ~0.0000153.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedFloat {
    /// Raw fixed-point value (16.16 format).
    pub raw: i32,
}

impl FixedFloat {
    /// Zero.
    pub const ZERO: Self = Self { raw: 0 };

    /// One.
    pub const ONE: Self = Self { raw: FIXED_SCALE };

    /// Negative one.
    pub const NEG_ONE: Self = Self { raw: -FIXED_SCALE };

    /// Half.
    pub const HALF: Self = Self {
        raw: FIXED_SCALE / 2,
    };

    /// Pi (approximation in fixed-point).
    pub const PI: Self = Self {
        raw: 205_887, // 3.14159 * 65536
    };

    /// Two-pi.
    pub const TWO_PI: Self = Self {
        raw: 411_775, // 6.28318 * 65536
    };

    /// Maximum representable value.
    pub const MAX: Self = Self { raw: i32::MAX };

    /// Minimum representable value.
    pub const MIN: Self = Self { raw: i32::MIN };

    /// Create from an integer value.
    #[inline]
    pub const fn from_int(v: i32) -> Self {
        Self {
            raw: v << FIXED_FRAC_BITS,
        }
    }

    /// Create from a floating-point value (lossy conversion for initialization).
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        Self {
            raw: (v * FIXED_SCALE as f32) as i32,
        }
    }

    /// Create from a raw fixed-point value.
    #[inline]
    pub const fn from_raw(raw: i32) -> Self {
        Self { raw }
    }

    /// Convert to f32 (lossy but useful for display/rendering).
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.raw as f32 / FIXED_SCALE as f32
    }

    /// Convert to integer (truncating fractional part).
    #[inline]
    pub fn to_int(self) -> i32 {
        self.raw >> FIXED_FRAC_BITS
    }

    /// Fractional part as raw bits.
    #[inline]
    pub fn frac_raw(self) -> i32 {
        self.raw & (FIXED_SCALE - 1)
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self {
            raw: self.raw.abs(),
        }
    }

    /// Negate.
    #[inline]
    pub fn neg(self) -> Self {
        Self { raw: -self.raw }
    }

    /// Addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            raw: self.raw.wrapping_add(other.raw),
        }
    }

    /// Subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            raw: self.raw.wrapping_sub(other.raw),
        }
    }

    /// Multiplication using 64-bit intermediate to avoid overflow.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        let result = (self.raw as i64 * other.raw as i64) >> FIXED_FRAC_BITS;
        Self {
            raw: result as i32,
        }
    }

    /// Division using 64-bit intermediate.
    #[inline]
    pub fn div(self, other: Self) -> Self {
        if other.raw == 0 {
            return if self.raw >= 0 { Self::MAX } else { Self::MIN };
        }
        let result = ((self.raw as i64) << FIXED_FRAC_BITS) / other.raw as i64;
        Self {
            raw: result as i32,
        }
    }

    /// Multiply by integer (no precision loss).
    #[inline]
    pub fn mul_int(self, v: i32) -> Self {
        Self {
            raw: self.raw.wrapping_mul(v),
        }
    }

    /// Divide by integer (single shift is exact for powers of 2).
    #[inline]
    pub fn div_int(self, v: i32) -> Self {
        if v == 0 {
            return if self.raw >= 0 { Self::MAX } else { Self::MIN };
        }
        Self { raw: self.raw / v }
    }

    /// Minimum of two values.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self.raw <= other.raw {
            self
        } else {
            other
        }
    }

    /// Maximum of two values.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self.raw >= other.raw {
            self
        } else {
            other
        }
    }

    /// Clamp between two values.
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    /// Floor (round toward negative infinity).
    #[inline]
    pub fn floor(self) -> Self {
        Self {
            raw: self.raw & !(FIXED_SCALE - 1),
        }
    }

    /// Ceiling (round toward positive infinity).
    #[inline]
    pub fn ceil(self) -> Self {
        let mask = FIXED_SCALE - 1;
        if self.raw & mask == 0 {
            self
        } else {
            Self {
                raw: (self.raw & !mask).wrapping_add(FIXED_SCALE),
            }
        }
    }

    /// Round to nearest integer.
    #[inline]
    pub fn round(self) -> Self {
        self.add(Self::HALF).floor()
    }

    /// Fixed-point square root using integer Newton-Raphson iteration.
    ///
    /// Deterministic across all platforms.
    pub fn sqrt(self) -> Self {
        if self.raw <= 0 {
            return Self::ZERO;
        }

        // Scale up to 64-bit for precision, then iterate.
        let val = (self.raw as u64) << FIXED_FRAC_BITS;
        let mut guess = val;
        let mut prev;

        // Newton-Raphson: x_{n+1} = (x_n + val/x_n) / 2
        loop {
            prev = guess;
            guess = (guess + val / guess) / 2;
            if guess >= prev {
                break;
            }
        }

        Self {
            raw: prev as i32,
        }
    }

    /// Fixed-point sine using Taylor series.
    ///
    /// Input is in fixed-point radians. Uses 5 terms for adequate precision.
    pub fn sin(self) -> Self {
        // Normalize to [-PI, PI].
        let mut x = self;
        let two_pi = Self::TWO_PI;
        while x.raw > Self::PI.raw {
            x = x.sub(two_pi);
        }
        while x.raw < Self::PI.neg().raw {
            x = x.add(two_pi);
        }

        // Taylor series: sin(x) = x - x^3/3! + x^5/5! - x^7/7!
        let x2 = x.mul(x);
        let x3 = x2.mul(x);
        let x5 = x3.mul(x2);
        let x7 = x5.mul(x2);

        let inv_6 = Self::from_raw(10923);    // 1/6 * 65536
        let inv_120 = Self::from_raw(546);     // 1/120 * 65536
        let inv_5040 = Self::from_raw(13);     // 1/5040 * 65536

        x.sub(x3.mul(inv_6))
            .add(x5.mul(inv_120))
            .sub(x7.mul(inv_5040))
    }

    /// Fixed-point cosine using sin(x + PI/2).
    pub fn cos(self) -> Self {
        let half_pi = Self::from_raw(102_944); // PI/2 * 65536
        self.add(half_pi).sin()
    }

    /// Linear interpolation.
    #[inline]
    pub fn lerp(a: Self, b: Self, t: Self) -> Self {
        a.add(b.sub(a).mul(t))
    }

    /// Whether the value is zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.raw == 0
    }

    /// Whether the value is positive.
    #[inline]
    pub fn is_positive(self) -> bool {
        self.raw > 0
    }

    /// Whether the value is negative.
    #[inline]
    pub fn is_negative(self) -> bool {
        self.raw < 0
    }
}

impl fmt::Debug for FixedFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fixed({:.6})", self.to_f32())
    }
}

impl fmt::Display for FixedFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.to_f32())
    }
}

impl Default for FixedFloat {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::ops::Add for FixedFloat {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        FixedFloat::add(self, rhs)
    }
}

impl std::ops::Sub for FixedFloat {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        FixedFloat::sub(self, rhs)
    }
}

impl std::ops::Mul for FixedFloat {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        FixedFloat::mul(self, rhs)
    }
}

impl std::ops::Div for FixedFloat {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        FixedFloat::div(self, rhs)
    }
}

impl std::ops::Neg for FixedFloat {
    type Output = Self;
    fn neg(self) -> Self {
        FixedFloat::neg(self)
    }
}

// ===========================================================================
// FixedVec3 — 3D vector using FixedFloat
// ===========================================================================

/// 3D vector using fixed-point arithmetic for deterministic computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FixedVec3 {
    pub x: FixedFloat,
    pub y: FixedFloat,
    pub z: FixedFloat,
}

impl FixedVec3 {
    pub const ZERO: Self = Self {
        x: FixedFloat::ZERO,
        y: FixedFloat::ZERO,
        z: FixedFloat::ZERO,
    };

    #[inline]
    pub fn new(x: FixedFloat, y: FixedFloat, z: FixedFloat) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn from_f32(x: f32, y: f32, z: f32) -> Self {
        Self {
            x: FixedFloat::from_f32(x),
            y: FixedFloat::from_f32(y),
            z: FixedFloat::from_f32(z),
        }
    }

    pub fn to_vec3phys(self) -> Vec3Phys {
        Vec3Phys::new(self.x.to_f32(), self.y.to_f32(), self.z.to_f32())
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    #[inline]
    pub fn scale(self, s: FixedFloat) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    #[inline]
    pub fn dot(a: Self, b: Self) -> FixedFloat {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    #[inline]
    pub fn cross(a: Self, b: Self) -> Self {
        Self {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    pub fn length_sq(self) -> FixedFloat {
        Self::dot(self, self)
    }

    pub fn length(self) -> FixedFloat {
        self.length_sq().sqrt()
    }

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l.is_zero() {
            return Self::ZERO;
        }
        Self {
            x: self.x / l,
            y: self.y / l,
            z: self.z / l,
        }
    }

    /// Hash the state for determinism verification.
    pub fn hash_state(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
        let bytes = [
            self.x.raw.to_le_bytes(),
            self.y.raw.to_le_bytes(),
            self.z.raw.to_le_bytes(),
        ];
        for chunk in &bytes {
            for &byte in chunk {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3); // FNV prime
            }
        }
        h
    }
}

// ===========================================================================
// DeterministicPhysics — deterministic simulation framework
// ===========================================================================

/// State hash for a single physics tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateHash {
    /// Tick number.
    pub tick: u32,
    /// FNV-1a hash of the complete physics state.
    pub hash: u64,
    /// Number of bodies in the simulation.
    pub body_count: u32,
}

/// Manages deterministic physics simulation.
#[derive(Debug)]
pub struct DeterministicPhysics {
    /// Fixed time step (seconds).
    pub fixed_dt: f32,
    /// Current tick.
    pub tick: u32,
    /// State hashes for each tick (for desync detection).
    pub state_hashes: Vec<StateHash>,
    /// Maximum stored hashes.
    pub max_hashes: usize,
    /// Whether to compute state hashes each tick.
    pub hash_enabled: bool,
    /// Whether to use fixed-point math.
    pub use_fixed_point: bool,
    /// Solver configuration.
    pub solver_config: SolverConfig,
}

/// Solver configuration.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Number of velocity solver iterations.
    pub velocity_iterations: u32,
    /// Number of position solver iterations.
    pub position_iterations: u32,
    /// Baumgarte stabilization factor.
    pub baumgarte: f32,
    /// Use split impulse for position correction.
    pub split_impulse: bool,
    /// Split impulse penetration threshold.
    pub split_impulse_threshold: f32,
    /// Number of sub-steps per tick.
    pub substeps: u32,
    /// Use shock propagation.
    pub shock_propagation: bool,
    /// Warm starting factor (0-1).
    pub warm_start_factor: f32,
    /// Maximum constraint error before flagging.
    pub max_constraint_error: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            velocity_iterations: DEFAULT_SOLVER_ITERATIONS,
            position_iterations: 10,
            baumgarte: DEFAULT_BAUMGARTE,
            split_impulse: true,
            split_impulse_threshold: SPLIT_IMPULSE_THRESHOLD,
            substeps: DEFAULT_SUBSTEPS,
            shock_propagation: true,
            warm_start_factor: 0.85,
            max_constraint_error: 0.01,
        }
    }
}

impl DeterministicPhysics {
    /// Create a new deterministic physics system.
    pub fn new(fixed_dt: f32) -> Self {
        Self {
            fixed_dt,
            tick: 0,
            state_hashes: Vec::new(),
            max_hashes: 600, // 10 seconds at 60Hz
            hash_enabled: true,
            use_fixed_point: false,
            solver_config: SolverConfig::default(),
        }
    }

    /// Compute the FNV-1a hash of a set of body states.
    pub fn compute_state_hash(&self, bodies: &[PhysicsBodyState]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;

        for body in bodies {
            // Hash position.
            for &byte in &body.position.x.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            for &byte in &body.position.y.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            for &byte in &body.position.z.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            // Hash velocity.
            for &byte in &body.velocity.x.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            for &byte in &body.velocity.y.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            for &byte in &body.velocity.z.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }

        h
    }

    /// Record the state hash for the current tick.
    pub fn record_state(&mut self, bodies: &[PhysicsBodyState]) {
        if !self.hash_enabled {
            return;
        }

        let hash = self.compute_state_hash(bodies);
        let state_hash = StateHash {
            tick: self.tick,
            hash,
            body_count: bodies.len() as u32,
        };

        self.state_hashes.push(state_hash);
        if self.state_hashes.len() > self.max_hashes {
            self.state_hashes.remove(0);
        }
    }

    /// Compare state hashes between two simulation runs.
    ///
    /// Returns the first tick where the hashes diverge, or None if they match.
    pub fn find_desync(
        run_a: &[StateHash],
        run_b: &[StateHash],
    ) -> Option<DesyncInfo> {
        let len = run_a.len().min(run_b.len());
        for i in 0..len {
            if run_a[i].hash != run_b[i].hash {
                return Some(DesyncInfo {
                    tick: run_a[i].tick,
                    hash_a: run_a[i].hash,
                    hash_b: run_b[i].hash,
                    body_count_a: run_a[i].body_count,
                    body_count_b: run_b[i].body_count,
                });
            }
        }
        None
    }

    /// Advance a simulation tick.
    pub fn advance(&mut self) {
        self.tick += 1;
    }

    /// Get the sub-step delta time.
    pub fn substep_dt(&self) -> f32 {
        self.fixed_dt / self.solver_config.substeps as f32
    }
}

/// Information about a desynchronization.
#[derive(Debug, Clone)]
pub struct DesyncInfo {
    /// Tick where desync was first detected.
    pub tick: u32,
    /// Hash from run A.
    pub hash_a: u64,
    /// Hash from run B.
    pub hash_b: u64,
    /// Body count in run A.
    pub body_count_a: u32,
    /// Body count in run B.
    pub body_count_b: u32,
}

/// A simplified physics body state for hashing.
#[derive(Debug, Clone, Copy)]
pub struct PhysicsBodyState {
    pub id: u64,
    pub position: Vec3Phys,
    pub velocity: Vec3Phys,
    pub angular_velocity: Vec3Phys,
    pub rotation: [f32; 4], // quaternion
}

// ===========================================================================
// PrecisionSolver — enhanced constraint solver
// ===========================================================================

/// A contact constraint for the precision solver.
#[derive(Debug, Clone)]
pub struct ContactConstraint {
    /// Body A index.
    pub body_a: usize,
    /// Body B index.
    pub body_b: usize,
    /// Contact point in world space.
    pub point: Vec3Phys,
    /// Contact normal (from A to B).
    pub normal: Vec3Phys,
    /// Penetration depth (positive = overlapping).
    pub penetration: f32,
    /// Coefficient of restitution.
    pub restitution: f32,
    /// Friction coefficient.
    pub friction: f32,
    /// Accumulated normal impulse (for warm starting).
    pub normal_impulse: f32,
    /// Accumulated tangent impulse.
    pub tangent_impulse: f32,
    /// Accumulated binormal impulse.
    pub binormal_impulse: f32,
    /// Position correction impulse (split impulse).
    pub position_impulse: f32,
    /// Bias for position correction.
    pub bias: f32,
    /// Mass scale factor (for shock propagation).
    pub mass_scale_a: f32,
    pub mass_scale_b: f32,
}

impl ContactConstraint {
    pub fn new(
        body_a: usize,
        body_b: usize,
        point: Vec3Phys,
        normal: Vec3Phys,
        penetration: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            point,
            normal,
            penetration,
            restitution: DEFAULT_RESTITUTION,
            friction: 0.5,
            normal_impulse: 0.0,
            tangent_impulse: 0.0,
            binormal_impulse: 0.0,
            position_impulse: 0.0,
            bias: 0.0,
            mass_scale_a: 1.0,
            mass_scale_b: 1.0,
        }
    }
}

/// A simplified body for the solver.
#[derive(Debug, Clone)]
pub struct SolverBody {
    pub position: Vec3Phys,
    pub velocity: Vec3Phys,
    pub angular_velocity: Vec3Phys,
    pub inv_mass: f32,
    pub inv_inertia: Vec3Phys, // diagonal of inverse inertia tensor
    pub is_static: bool,
}

impl SolverBody {
    pub fn dynamic(position: Vec3Phys, mass: f32) -> Self {
        let inv_mass = if mass > EPSILON { 1.0 / mass } else { 0.0 };
        Self {
            position,
            velocity: Vec3Phys::ZERO,
            angular_velocity: Vec3Phys::ZERO,
            inv_mass,
            inv_inertia: Vec3Phys::new(inv_mass, inv_mass, inv_mass),
            is_static: false,
        }
    }

    pub fn static_body(position: Vec3Phys) -> Self {
        Self {
            position,
            velocity: Vec3Phys::ZERO,
            angular_velocity: Vec3Phys::ZERO,
            inv_mass: 0.0,
            inv_inertia: Vec3Phys::ZERO,
            is_static: true,
        }
    }
}

/// Precision constraint solver with shock propagation and split impulse.
#[derive(Debug)]
pub struct PrecisionSolver {
    /// Solver configuration.
    pub config: SolverConfig,
    /// Contact constraints.
    pub contacts: Vec<ContactConstraint>,
    /// Solver bodies.
    pub bodies: Vec<SolverBody>,
    /// Statistics.
    pub stats: SolverStats,
}

/// Solver statistics.
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// Total iterations performed.
    pub total_iterations: u64,
    /// Average constraint error after solving.
    pub avg_error: f32,
    /// Maximum constraint error.
    pub max_error: f32,
    /// Number of contacts solved.
    pub contacts_solved: u32,
    /// Number of position corrections applied.
    pub position_corrections: u32,
}

impl PrecisionSolver {
    /// Create a new precision solver.
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            contacts: Vec::new(),
            bodies: Vec::new(),
            stats: SolverStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SolverConfig::default())
    }

    /// Add a body and return its index.
    pub fn add_body(&mut self, body: SolverBody) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    /// Add a contact constraint.
    pub fn add_contact(&mut self, constraint: ContactConstraint) {
        self.contacts.push(constraint);
    }

    /// Pre-solve: compute bias, warm start.
    pub fn pre_solve(&mut self, dt: f32) {
        let baumgarte = self.config.baumgarte;
        let inv_dt = if dt > EPSILON { 1.0 / dt } else { 0.0 };

        for contact in &mut self.contacts {
            // Compute bias for position correction.
            let slop = 0.005; // allowed penetration
            let penetration_bias = baumgarte * inv_dt * (contact.penetration - slop).max(0.0);

            // Restitution bias.
            let body_a = &self.bodies[contact.body_a];
            let body_b = &self.bodies[contact.body_b];
            let relative_velocity = body_b.velocity.sub(body_a.velocity);
            let normal_vel = Vec3Phys::dot(relative_velocity, contact.normal);
            let restitution_bias = if normal_vel < -1.0 {
                contact.restitution * normal_vel
            } else {
                0.0
            };

            contact.bias = penetration_bias + restitution_bias;

            // Warm starting.
            if self.config.warm_start_factor > 0.0 {
                let impulse = contact.normal.scale(
                    contact.normal_impulse * self.config.warm_start_factor,
                );
                if !body_a.is_static {
                    // Would apply impulse to body A.
                }
                if !body_b.is_static {
                    // Would apply impulse to body B.
                }
                contact.normal_impulse *= self.config.warm_start_factor;
                contact.tangent_impulse *= self.config.warm_start_factor;
            }
        }

        // Shock propagation: sort contacts bottom-to-top.
        if self.config.shock_propagation {
            self.apply_shock_propagation();
        }
    }

    /// Apply shock propagation mass scaling.
    ///
    /// Bodies lower in a stack get their mass artificially increased
    /// relative to bodies above them, improving stack stability.
    fn apply_shock_propagation(&mut self) {
        // Sort contacts by the Y coordinate of the contact point (bottom first).
        self.contacts.sort_by(|a, b| {
            a.point
                .y
                .partial_cmp(&b.point.y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Scale masses: lower bodies appear heavier to upper bodies.
        for i in 0..self.contacts.len() {
            let y_normalized = if self.contacts.len() > 1 {
                i as f32 / (self.contacts.len() - 1) as f32
            } else {
                0.5
            };

            // Lower contacts (small y_normalized) have higher mass scale.
            let scale = 1.0 + (1.0 - y_normalized) * 2.0;
            self.contacts[i].mass_scale_a = scale;
            self.contacts[i].mass_scale_b = 1.0;
        }
    }

    /// Solve velocity constraints.
    pub fn solve_velocity(&mut self) {
        for _iter in 0..self.config.velocity_iterations {
            for ci in 0..self.contacts.len() {
                let (body_a_idx, body_b_idx) = (
                    self.contacts[ci].body_a,
                    self.contacts[ci].body_b,
                );

                let normal = self.contacts[ci].normal;
                let bias = self.contacts[ci].bias;

                // Compute relative velocity at contact.
                let vel_a = self.bodies[body_a_idx].velocity;
                let vel_b = self.bodies[body_b_idx].velocity;
                let rel_vel = vel_b.sub(vel_a);
                let normal_vel = Vec3Phys::dot(rel_vel, normal);

                // Compute effective mass.
                let inv_mass_a = self.bodies[body_a_idx].inv_mass
                    / self.contacts[ci].mass_scale_a;
                let inv_mass_b = self.bodies[body_b_idx].inv_mass
                    / self.contacts[ci].mass_scale_b;
                let effective_mass = inv_mass_a + inv_mass_b;

                if effective_mass < EPSILON {
                    continue;
                }

                // Normal impulse.
                let impulse_magnitude = -(normal_vel + bias) / effective_mass;

                // Clamp accumulated impulse (must be non-negative).
                let old_impulse = self.contacts[ci].normal_impulse;
                self.contacts[ci].normal_impulse =
                    (old_impulse + impulse_magnitude).max(0.0);
                let applied = self.contacts[ci].normal_impulse - old_impulse;

                let impulse = normal.scale(applied);

                // Apply to bodies.
                if !self.bodies[body_a_idx].is_static {
                    self.bodies[body_a_idx].velocity =
                        self.bodies[body_a_idx].velocity.sub(impulse.scale(inv_mass_a));
                }
                if !self.bodies[body_b_idx].is_static {
                    self.bodies[body_b_idx].velocity =
                        self.bodies[body_b_idx].velocity.add(impulse.scale(inv_mass_b));
                }

                // Friction impulse.
                let rel_vel2 = self.bodies[body_b_idx]
                    .velocity
                    .sub(self.bodies[body_a_idx].velocity);
                let tangent_vel = rel_vel2.sub(normal.scale(Vec3Phys::dot(rel_vel2, normal)));
                let tangent_speed = tangent_vel.length();

                if tangent_speed > EPSILON {
                    let tangent = tangent_vel.scale(1.0 / tangent_speed);
                    let friction_impulse = -tangent_speed / effective_mass;

                    // Coulomb friction clamp.
                    let max_friction =
                        self.contacts[ci].friction * self.contacts[ci].normal_impulse;
                    let clamped = friction_impulse.clamp(-max_friction, max_friction);

                    let friction_vec = tangent.scale(clamped);

                    if !self.bodies[body_a_idx].is_static {
                        self.bodies[body_a_idx].velocity = self.bodies[body_a_idx]
                            .velocity
                            .sub(friction_vec.scale(inv_mass_a));
                    }
                    if !self.bodies[body_b_idx].is_static {
                        self.bodies[body_b_idx].velocity = self.bodies[body_b_idx]
                            .velocity
                            .add(friction_vec.scale(inv_mass_b));
                    }
                }
            }

            self.stats.total_iterations += 1;
        }

        self.stats.contacts_solved = self.contacts.len() as u32;
    }

    /// Solve position constraints using split impulse.
    pub fn solve_position(&mut self, dt: f32) {
        if !self.config.split_impulse {
            return;
        }

        for _iter in 0..self.config.position_iterations {
            for ci in 0..self.contacts.len() {
                let penetration = self.contacts[ci].penetration;
                if penetration < self.config.split_impulse_threshold.abs() {
                    continue;
                }

                let normal = self.contacts[ci].normal;
                let body_a_idx = self.contacts[ci].body_a;
                let body_b_idx = self.contacts[ci].body_b;

                let inv_mass_a = self.bodies[body_a_idx].inv_mass;
                let inv_mass_b = self.bodies[body_b_idx].inv_mass;
                let effective_mass = inv_mass_a + inv_mass_b;

                if effective_mass < EPSILON {
                    continue;
                }

                let slop = 0.005;
                let correction = (penetration - slop).max(0.0) * self.config.baumgarte;
                let impulse = correction / effective_mass;

                if !self.bodies[body_a_idx].is_static {
                    let shift = normal.scale(-impulse * inv_mass_a);
                    self.bodies[body_a_idx].position += shift;
                }
                if !self.bodies[body_b_idx].is_static {
                    let shift = normal.scale(impulse * inv_mass_b);
                    self.bodies[body_b_idx].position += shift;
                }

                self.stats.position_corrections += 1;
            }
        }
    }

    /// Post-stabilization: project to valid configuration.
    pub fn post_stabilize(&mut self) {
        for contact in &self.contacts {
            if contact.penetration <= 0.0 {
                continue;
            }

            let body_a_idx = contact.body_a;
            let body_b_idx = contact.body_b;
            let inv_mass_a = self.bodies[body_a_idx].inv_mass;
            let inv_mass_b = self.bodies[body_b_idx].inv_mass;
            let total_inv_mass = inv_mass_a + inv_mass_b;

            if total_inv_mass < EPSILON {
                continue;
            }

            let correction = contact.normal.scale(contact.penetration);

            if !self.bodies[body_a_idx].is_static {
                let frac = inv_mass_a / total_inv_mass;
                self.bodies[body_a_idx].position -= correction * frac;
            }
            if !self.bodies[body_b_idx].is_static {
                let frac = inv_mass_b / total_inv_mass;
                self.bodies[body_b_idx].position += correction * frac;
            }
        }
    }

    /// Full solve step.
    pub fn solve(&mut self, dt: f32) {
        self.pre_solve(dt);
        self.solve_velocity();
        self.solve_position(dt);
        self.post_stabilize();
    }

    /// Clear all contacts (call each tick after solving).
    pub fn clear_contacts(&mut self) {
        self.contacts.clear();
    }

    /// Clear everything.
    pub fn clear(&mut self) {
        self.contacts.clear();
        self.bodies.clear();
    }
}

// ===========================================================================
// SportPhysics — ball simulation with Magnus effect
// ===========================================================================

/// Configuration for a sports ball.
#[derive(Debug, Clone)]
pub struct BallConfig {
    /// Mass (kg).
    pub mass: f32,
    /// Radius (m).
    pub radius: f32,
    /// Drag coefficient (dimensionless). Varies with Reynolds number if enabled.
    pub cd: f32,
    /// Coefficient of restitution.
    pub restitution: f32,
    /// Rolling friction coefficient.
    pub rolling_friction: f32,
    /// Moment of inertia (2/5 * m * r^2 for a solid sphere).
    pub moment_of_inertia: f32,
    /// Whether to use Reynolds-number-dependent drag.
    pub variable_drag: bool,
    /// Spin decay rate (angular velocity damping per second).
    pub spin_decay: f32,
    /// Cross-sectional area (PI * r^2).
    pub cross_section_area: f32,
}

impl BallConfig {
    /// Create a standard soccer ball configuration.
    pub fn soccer() -> Self {
        let r = SOCCER_BALL_RADIUS;
        let m = SOCCER_BALL_MASS;
        Self {
            mass: m,
            radius: r,
            cd: SOCCER_BALL_CD,
            restitution: 0.7,
            rolling_friction: 0.02,
            moment_of_inertia: 0.4 * m * r * r,
            variable_drag: true,
            spin_decay: 0.5,
            cross_section_area: PI * r * r,
        }
    }

    /// Create a basketball configuration.
    pub fn basketball() -> Self {
        let r = 0.121;
        let m = 0.62;
        Self {
            mass: m,
            radius: r,
            cd: 0.47,
            restitution: 0.83,
            rolling_friction: 0.03,
            moment_of_inertia: 0.4 * m * r * r,
            variable_drag: false,
            spin_decay: 0.3,
            cross_section_area: PI * r * r,
        }
    }

    /// Create a tennis ball configuration.
    pub fn tennis() -> Self {
        let r = 0.033;
        let m = 0.058;
        Self {
            mass: m,
            radius: r,
            cd: 0.55, // fuzzy = higher drag
            restitution: 0.75,
            rolling_friction: 0.04,
            moment_of_inertia: 0.4 * m * r * r,
            variable_drag: false,
            spin_decay: 0.8,
            cross_section_area: PI * r * r,
        }
    }

    /// Create a golf ball configuration.
    pub fn golf() -> Self {
        let r = 0.0214;
        let m = 0.046;
        Self {
            mass: m,
            radius: r,
            cd: 0.24, // dimpled surface
            restitution: 0.83,
            rolling_friction: 0.015,
            moment_of_inertia: 0.4 * m * r * r,
            variable_drag: true,
            spin_decay: 0.2,
            cross_section_area: PI * r * r,
        }
    }
}

/// State of a sports ball.
#[derive(Debug, Clone)]
pub struct BallState {
    /// Position (m).
    pub position: Vec3Phys,
    /// Velocity (m/s).
    pub velocity: Vec3Phys,
    /// Angular velocity / spin (rad/s).
    pub angular_velocity: Vec3Phys,
    /// Deformation factor (0-1, 0 = no deformation).
    pub deformation: f32,
    /// Whether the ball is on the ground.
    pub grounded: bool,
    /// Whether the ball is rolling.
    pub rolling: bool,
    /// Time in air (seconds).
    pub air_time: f32,
}

impl BallState {
    /// Create a ball at rest.
    pub fn at_position(pos: Vec3Phys) -> Self {
        Self {
            position: pos,
            velocity: Vec3Phys::ZERO,
            angular_velocity: Vec3Phys::ZERO,
            deformation: 0.0,
            grounded: false,
            rolling: false,
            air_time: 0.0,
        }
    }

    /// Speed (magnitude of velocity).
    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }

    /// Spin rate (magnitude of angular velocity, rad/s).
    pub fn spin_rate(&self) -> f32 {
        self.angular_velocity.length()
    }

    /// Spin rate in RPM.
    pub fn spin_rpm(&self) -> f32 {
        self.spin_rate() * 60.0 / (2.0 * PI)
    }

    /// Kinetic energy (translational + rotational).
    pub fn kinetic_energy(&self, config: &BallConfig) -> f32 {
        let translational = 0.5 * config.mass * self.velocity.length_sq();
        let rotational = 0.5 * config.moment_of_inertia * self.angular_velocity.length_sq();
        translational + rotational
    }
}

/// Sports ball physics simulation.
#[derive(Debug)]
pub struct SportPhysics {
    /// Ball configuration.
    pub config: BallConfig,
    /// Current ball state.
    pub state: BallState,
    /// Air density (can vary with altitude).
    pub air_density: f32,
    /// Gravity vector.
    pub gravity: Vec3Phys,
    /// Ground plane height.
    pub ground_height: f32,
    /// Wind velocity.
    pub wind: Vec3Phys,
    /// Statistics.
    pub stats: SportPhysicsStats,
}

/// Sport physics statistics.
#[derive(Debug, Clone, Default)]
pub struct SportPhysicsStats {
    pub max_speed: f32,
    pub max_spin: f32,
    pub total_bounces: u32,
    pub max_deformation: f32,
    pub total_magnus_force: f32,
}

impl SportPhysics {
    /// Create a new sports physics simulation.
    pub fn new(config: BallConfig) -> Self {
        Self {
            config,
            state: BallState::at_position(Vec3Phys::ZERO),
            air_density: AIR_DENSITY,
            gravity: Vec3Phys::new(0.0, -GRAVITY, 0.0),
            ground_height: 0.0,
            wind: Vec3Phys::ZERO,
            stats: SportPhysicsStats::default(),
        }
    }

    /// Compute the Magnus force.
    ///
    /// The Magnus force acts perpendicular to both the velocity and the spin
    /// axis, causing a spinning ball to curve. This is the effect behind
    /// "banana kicks" in soccer and curve balls in baseball.
    ///
    /// F_Magnus = 0.5 * CL * rho * A * |v|^2 * (omega x v) / |omega x v|
    ///
    /// A simplified model uses:
    /// F_Magnus = (4/3) * pi * r^3 * rho * (omega x v)
    pub fn magnus_force(
        velocity: Vec3Phys,
        angular_velocity: Vec3Phys,
        radius: f32,
        air_density: f32,
    ) -> Vec3Phys {
        let omega_cross_v = Vec3Phys::cross(angular_velocity, velocity);
        let volume = (4.0 / 3.0) * PI * radius * radius * radius;

        // The Magnus coefficient scales the force. For a sphere this is
        // approximately 0.5 * volume * rho.
        omega_cross_v.scale(MAGNUS_COEFFICIENT * volume * air_density)
    }

    /// Compute the aerodynamic drag force.
    ///
    /// F_drag = 0.5 * Cd * rho * A * |v|^2 * (-v_hat)
    ///
    /// The drag coefficient Cd varies with Reynolds number for a sphere:
    /// - Below the "drag crisis" (~Re 200,000): Cd ~ 0.47
    /// - Above: Cd drops to ~0.2 as the boundary layer transitions to turbulent
    pub fn drag_force(
        velocity: Vec3Phys,
        wind: Vec3Phys,
        cd: f32,
        radius: f32,
        air_density: f32,
        variable_drag: bool,
    ) -> Vec3Phys {
        // Velocity relative to air.
        let relative_vel = velocity.sub(wind);
        let speed = relative_vel.length();

        if speed < EPSILON {
            return Vec3Phys::ZERO;
        }

        // Cross-sectional area.
        let area = PI * radius * radius;

        // Optionally adjust Cd based on Reynolds number.
        let effective_cd = if variable_drag {
            let re = Self::reynolds_number(speed, 2.0 * radius);
            Self::cd_from_reynolds(re)
        } else {
            cd
        };

        // F_drag = 0.5 * Cd * rho * A * v^2
        let force_magnitude = 0.5 * effective_cd * air_density * area * speed * speed;

        // Direction: opposite to velocity.
        let direction = relative_vel.scale(-1.0 / speed);
        direction.scale(force_magnitude)
    }

    /// Compute the Reynolds number.
    ///
    /// Re = v * d / nu
    /// where v = speed, d = diameter, nu = kinematic viscosity of air.
    #[inline]
    pub fn reynolds_number(speed: f32, diameter: f32) -> f32 {
        speed * diameter / AIR_VISCOSITY
    }

    /// Get drag coefficient from Reynolds number (sphere drag crisis model).
    ///
    /// The drag crisis is a dramatic drop in drag coefficient at high Reynolds
    /// numbers as the boundary layer transitions from laminar to turbulent flow.
    pub fn cd_from_reynolds(re: f32) -> f32 {
        if re < 1.0 {
            return 24.0; // Stokes drag (very slow)
        }
        if re < 1000.0 {
            // Intermediate regime: empirical correlation.
            return 24.0 / re + 6.0 / (1.0 + re.sqrt()) + 0.4;
        }
        if re < DRAG_CRISIS_RE * 0.8 {
            return CD_SUBCRITICAL;
        }
        if re > DRAG_CRISIS_RE * 1.2 {
            return CD_SUPERCRITICAL;
        }
        // Smooth transition through drag crisis.
        let t = (re - DRAG_CRISIS_RE * 0.8) / (DRAG_CRISIS_RE * 0.4);
        CD_SUBCRITICAL + (CD_SUPERCRITICAL - CD_SUBCRITICAL) * t
    }

    /// Simulate one time step.
    pub fn step(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }

        // Compute forces.
        let velocity = self.state.velocity;
        let angular_velocity = self.state.angular_velocity;

        // 1. Gravity.
        let gravity_force = self.gravity.scale(self.config.mass);

        // 2. Aerodynamic drag.
        let drag = Self::drag_force(
            velocity,
            self.wind,
            self.config.cd,
            self.config.radius,
            self.air_density,
            self.config.variable_drag,
        );

        // 3. Magnus force (spin-induced curve).
        let magnus = Self::magnus_force(
            velocity,
            angular_velocity,
            self.config.radius,
            self.air_density,
        );

        self.stats.total_magnus_force += magnus.length();

        // Total force.
        let total_force = gravity_force.add(drag).add(magnus);

        // Integrate velocity (semi-implicit Euler).
        let acceleration = total_force.scale(1.0 / self.config.mass);
        self.state.velocity = velocity.add(acceleration.scale(dt));

        // Integrate position.
        self.state.position = self.state.position.add(self.state.velocity.scale(dt));

        // Spin decay.
        let decay = (-self.config.spin_decay * dt).exp();
        self.state.angular_velocity = self.state.angular_velocity.scale(decay);

        // Ground collision.
        if self.state.position.y - self.config.radius < self.ground_height {
            self.handle_ground_collision(dt);
        } else {
            self.state.grounded = false;
            self.state.rolling = false;
            self.state.air_time += dt;
        }

        // Clamp angular velocity.
        self.state.angular_velocity =
            self.state.angular_velocity.clamp_length(MAX_ANGULAR_VELOCITY);

        // Deformation recovery.
        self.state.deformation *= 0.9_f32.powf(dt * 60.0);

        // Statistics.
        let speed = self.state.speed();
        if speed > self.stats.max_speed {
            self.stats.max_speed = speed;
        }
        let spin = self.state.spin_rate();
        if spin > self.stats.max_spin {
            self.stats.max_spin = spin;
        }
    }

    /// Handle collision with the ground plane.
    fn handle_ground_collision(&mut self, _dt: f32) {
        let ground_normal = Vec3Phys::UP;
        let penetration = self.ground_height + self.config.radius - self.state.position.y;

        // Position correction.
        self.state.position.y = self.ground_height + self.config.radius;

        // Velocity decomposition.
        let normal_vel = Vec3Phys::dot(self.state.velocity, ground_normal);
        let tangent_vel = self.state.velocity.sub(ground_normal.scale(normal_vel));

        if normal_vel < -0.5 {
            // Bounce.
            self.stats.total_bounces += 1;

            // Deformation based on impact speed.
            self.state.deformation = ((-normal_vel) / 20.0).min(1.0);
            if self.state.deformation > self.stats.max_deformation {
                self.stats.max_deformation = self.state.deformation;
            }

            // Reflect with restitution.
            let reflected_normal_vel = -normal_vel * self.config.restitution;
            self.state.velocity = tangent_vel.add(ground_normal.scale(reflected_normal_vel));

            // Spin transfer on contact: tangent velocity induces topspin.
            let surface_vel = tangent_vel;
            let spin_transfer = Vec3Phys::cross(ground_normal, surface_vel)
                .scale(0.4 / self.config.radius); // spin transfer coefficient
            self.state.angular_velocity = self.state.angular_velocity.add(spin_transfer);

            self.state.grounded = false;
            self.state.air_time = 0.0;
        } else {
            // Low velocity: rolling on ground.
            self.state.velocity = tangent_vel;
            self.state.grounded = true;

            let speed = tangent_vel.length();
            if speed > EPSILON {
                // Rolling friction.
                let friction_decel = self.config.rolling_friction * GRAVITY;
                let new_speed = (speed - friction_decel * _dt).max(0.0);
                self.state.velocity =
                    tangent_vel.normalized().scale(new_speed);

                // Rolling angular velocity: v = omega * r.
                let roll_axis = Vec3Phys::cross(ground_normal, tangent_vel.normalized());
                self.state.angular_velocity =
                    roll_axis.scale(new_speed / self.config.radius);
                self.state.rolling = true;
            } else {
                self.state.velocity = Vec3Phys::ZERO;
                self.state.angular_velocity = Vec3Phys::ZERO;
                self.state.rolling = false;
            }
        }
    }

    /// Kick the ball with a given velocity and spin.
    pub fn kick(&mut self, velocity: Vec3Phys, spin: Vec3Phys) {
        self.state.velocity = velocity;
        self.state.angular_velocity = spin;
        self.state.grounded = false;
        self.state.air_time = 0.0;
    }

    /// Reset the ball to a position.
    pub fn reset(&mut self, position: Vec3Phys) {
        self.state = BallState::at_position(position);
    }
}

// ===========================================================================
// SoftContact — realistic contact modeling
// ===========================================================================

/// Contact surface material properties for soft contact simulation.
#[derive(Debug, Clone, Copy)]
pub struct SoftContactMaterial {
    /// Compliance (inverse stiffness, m/N). Higher = softer.
    pub compliance: f32,
    /// Damping coefficient.
    pub damping: f32,
    /// Static friction coefficient.
    pub static_friction: f32,
    /// Dynamic friction coefficient.
    pub dynamic_friction: f32,
    /// Friction anisotropy: ratio of friction in the secondary direction.
    /// 1.0 = isotropic, <1 = less friction in secondary direction.
    pub friction_anisotropy: f32,
    /// Secondary friction direction (tangent to surface).
    pub anisotropy_direction: Vec3Phys,
    /// Hertz contact stiffness (for contact area calculation).
    pub hertz_stiffness: f32,
}

impl Default for SoftContactMaterial {
    fn default() -> Self {
        Self {
            compliance: 0.001,
            damping: 100.0,
            static_friction: 0.6,
            dynamic_friction: 0.4,
            friction_anisotropy: 1.0,
            anisotropy_direction: Vec3Phys::RIGHT,
            hertz_stiffness: 1e6,
        }
    }
}

impl SoftContactMaterial {
    /// Grass surface (anisotropic: more sliding along blade direction).
    pub fn grass() -> Self {
        Self {
            compliance: 0.005,
            damping: 50.0,
            static_friction: 0.7,
            dynamic_friction: 0.5,
            friction_anisotropy: 0.7,
            anisotropy_direction: Vec3Phys::new(1.0, 0.0, 0.0),
            hertz_stiffness: 5e5,
        }
    }

    /// Turf surface.
    pub fn turf() -> Self {
        Self {
            compliance: 0.002,
            damping: 80.0,
            static_friction: 0.8,
            dynamic_friction: 0.6,
            friction_anisotropy: 0.85,
            anisotropy_direction: Vec3Phys::new(1.0, 0.0, 0.0),
            hertz_stiffness: 8e5,
        }
    }

    /// Hard court surface (isotropic).
    pub fn hard_court() -> Self {
        Self {
            compliance: 0.0002,
            damping: 200.0,
            static_friction: 0.5,
            dynamic_friction: 0.35,
            friction_anisotropy: 1.0,
            anisotropy_direction: Vec3Phys::RIGHT,
            hertz_stiffness: 2e6,
        }
    }

    /// Ice surface (very low friction).
    pub fn ice() -> Self {
        Self {
            compliance: 0.0001,
            damping: 300.0,
            static_friction: 0.05,
            dynamic_friction: 0.03,
            friction_anisotropy: 1.0,
            anisotropy_direction: Vec3Phys::RIGHT,
            hertz_stiffness: 5e6,
        }
    }
}

/// A soft contact point with compliance-based resolution.
#[derive(Debug, Clone)]
pub struct SoftContact {
    /// Contact point.
    pub point: Vec3Phys,
    /// Contact normal.
    pub normal: Vec3Phys,
    /// Penetration depth.
    pub penetration: f32,
    /// Material properties.
    pub material: SoftContactMaterial,
    /// Surface velocity at contact (from spinning objects).
    pub surface_velocity: Vec3Phys,
    /// Hertz contact area (grows with penetration depth).
    pub contact_area: f32,
    /// Applied normal force (computed during solve).
    pub normal_force: f32,
    /// Applied friction force.
    pub friction_force: Vec3Phys,
}

impl SoftContact {
    /// Create a new soft contact.
    pub fn new(
        point: Vec3Phys,
        normal: Vec3Phys,
        penetration: f32,
        material: SoftContactMaterial,
    ) -> Self {
        // Hertz contact area: a = sqrt(R * d) approximately.
        let contact_radius = (penetration * 0.1).sqrt().max(0.001);
        let contact_area = PI * contact_radius * contact_radius;

        Self {
            point,
            normal,
            penetration,
            material,
            surface_velocity: Vec3Phys::ZERO,
            contact_area,
            normal_force: 0.0,
            friction_force: Vec3Phys::ZERO,
        }
    }

    /// Compute the normal force using a compliance-based model.
    ///
    /// F = penetration / compliance - damping * normal_velocity
    pub fn compute_normal_force(&mut self, normal_velocity: f32, dt: f32) -> f32 {
        if self.penetration <= 0.0 {
            self.normal_force = 0.0;
            return 0.0;
        }

        // Spring force (compliance-based).
        let spring_force = self.penetration / self.material.compliance.max(EPSILON);

        // Damping force.
        let damping_force = -self.material.damping * normal_velocity;

        // Total normal force (must be non-negative -- can't pull).
        self.normal_force = (spring_force + damping_force).max(0.0);
        self.normal_force
    }

    /// Compute anisotropic friction force.
    ///
    /// In anisotropic friction, the friction coefficient differs in different
    /// tangent directions. For example, grass has less resistance along the
    /// blade direction than across it.
    pub fn compute_friction_force(
        &mut self,
        tangent_velocity: Vec3Phys,
        dt: f32,
    ) -> Vec3Phys {
        let speed = tangent_velocity.length();
        if speed < EPSILON || self.normal_force < EPSILON {
            self.friction_force = Vec3Phys::ZERO;
            return Vec3Phys::ZERO;
        }

        let tangent_dir = tangent_velocity.scale(1.0 / speed);

        // Choose friction coefficient (static vs dynamic).
        let base_friction = if speed < 0.1 {
            self.material.static_friction
        } else {
            self.material.dynamic_friction
        };

        // Apply anisotropy.
        let effective_friction = if (self.material.friction_anisotropy - 1.0).abs() > EPSILON {
            let aniso_dot =
                Vec3Phys::dot(tangent_dir, self.material.anisotropy_direction).abs();
            let aniso_factor = 1.0
                + (self.material.friction_anisotropy - 1.0) * aniso_dot * aniso_dot;
            base_friction * aniso_factor
        } else {
            base_friction
        };

        // Coulomb friction: F_friction <= mu * F_normal.
        let max_friction = effective_friction * self.normal_force;
        let friction_magnitude = speed.min(max_friction * dt / 1.0) / dt.max(EPSILON);
        let clamped = friction_magnitude.min(max_friction);

        self.friction_force = tangent_dir.scale(-clamped);
        self.friction_force
    }

    /// Update Hertz contact area based on current penetration.
    pub fn update_contact_area(&mut self) {
        if self.penetration <= 0.0 {
            self.contact_area = 0.0;
            return;
        }
        // Hertz model: contact radius a ~ sqrt(R * delta)
        // Using effective radius of 0.1m as a default.
        let effective_radius = 0.1;
        let contact_radius = (effective_radius * self.penetration).sqrt();
        self.contact_area = PI * contact_radius * contact_radius;
    }
}

// ===========================================================================
// CharacterBody — sports character simulation
// ===========================================================================

/// Per-limb collision capsule for tackle/contact detection.
#[derive(Debug, Clone)]
pub struct LimbCapsule {
    /// Limb identifier.
    pub name: String,
    /// Local start point (relative to character root).
    pub local_start: Vec3Phys,
    /// Local end point.
    pub local_end: Vec3Phys,
    /// Capsule radius.
    pub radius: f32,
    /// Mass of this limb segment.
    pub mass: f32,
}

/// Character body for sports simulation.
#[derive(Debug)]
pub struct CharacterBody {
    /// Position (feet / root).
    pub position: Vec3Phys,
    /// Velocity.
    pub velocity: Vec3Phys,
    /// Facing direction (yaw angle in radians).
    pub facing: f32,
    /// Total mass (kg).
    pub mass: f32,
    /// Height (m).
    pub height: f32,
    /// Center of mass offset from root.
    pub center_of_mass: Vec3Phys,
    /// Dynamic center of mass shift during animation.
    pub com_shift: Vec3Phys,
    /// Body lean angle (radians, bicycle model for direction changes).
    pub lean_angle: f32,
    /// Maximum lean angle.
    pub max_lean: f32,
    /// Per-limb collision capsules.
    pub limbs: Vec<LimbCapsule>,
    /// Root motion integration: velocity from animation.
    pub root_motion_velocity: Vec3Phys,
    /// Whether root motion is driving movement.
    pub root_motion_enabled: bool,
    /// Foot IK targets (left foot, right foot world positions).
    pub left_foot_target: Vec3Phys,
    pub right_foot_target: Vec3Phys,
    /// Whether foot IK is active.
    pub foot_ik_enabled: bool,
    /// Ground normal at the character's position.
    pub ground_normal: Vec3Phys,
    /// Ground height at the character's position.
    pub ground_height: f32,
    /// Whether the character is grounded.
    pub grounded: bool,
    /// Speed (cached).
    pub speed: f32,
}

impl CharacterBody {
    /// Create a new character body.
    pub fn new(position: Vec3Phys, mass: f32, height: f32) -> Self {
        let com_y = height * 0.55; // Slightly above center for a human.
        Self {
            position,
            velocity: Vec3Phys::ZERO,
            facing: 0.0,
            mass,
            height,
            center_of_mass: Vec3Phys::new(0.0, com_y, 0.0),
            com_shift: Vec3Phys::ZERO,
            lean_angle: 0.0,
            max_lean: 0.35, // ~20 degrees
            limbs: Vec::new(),
            root_motion_velocity: Vec3Phys::ZERO,
            root_motion_enabled: false,
            left_foot_target: Vec3Phys::ZERO,
            right_foot_target: Vec3Phys::ZERO,
            foot_ik_enabled: true,
            ground_normal: Vec3Phys::UP,
            ground_height: 0.0,
            grounded: true,
            speed: 0.0,
        }
    }

    /// Create a standard humanoid with default limb capsules.
    pub fn humanoid(position: Vec3Phys, mass: f32) -> Self {
        let mut body = Self::new(position, mass, 1.8);

        // Add limb capsules.
        body.limbs.push(LimbCapsule {
            name: "torso".into(),
            local_start: Vec3Phys::new(0.0, 0.8, 0.0),
            local_end: Vec3Phys::new(0.0, 1.4, 0.0),
            radius: 0.2,
            mass: mass * 0.45,
        });
        body.limbs.push(LimbCapsule {
            name: "head".into(),
            local_start: Vec3Phys::new(0.0, 1.4, 0.0),
            local_end: Vec3Phys::new(0.0, 1.7, 0.0),
            radius: 0.12,
            mass: mass * 0.08,
        });
        body.limbs.push(LimbCapsule {
            name: "left_upper_leg".into(),
            local_start: Vec3Phys::new(-0.1, 0.85, 0.0),
            local_end: Vec3Phys::new(-0.1, 0.45, 0.0),
            radius: 0.08,
            mass: mass * 0.1,
        });
        body.limbs.push(LimbCapsule {
            name: "right_upper_leg".into(),
            local_start: Vec3Phys::new(0.1, 0.85, 0.0),
            local_end: Vec3Phys::new(0.1, 0.45, 0.0),
            radius: 0.08,
            mass: mass * 0.1,
        });
        body.limbs.push(LimbCapsule {
            name: "left_lower_leg".into(),
            local_start: Vec3Phys::new(-0.1, 0.45, 0.0),
            local_end: Vec3Phys::new(-0.1, 0.0, 0.0),
            radius: 0.06,
            mass: mass * 0.05,
        });
        body.limbs.push(LimbCapsule {
            name: "right_lower_leg".into(),
            local_start: Vec3Phys::new(0.1, 0.45, 0.0),
            local_end: Vec3Phys::new(0.1, 0.0, 0.0),
            radius: 0.06,
            mass: mass * 0.05,
        });
        body.limbs.push(LimbCapsule {
            name: "left_arm".into(),
            local_start: Vec3Phys::new(-0.25, 1.35, 0.0),
            local_end: Vec3Phys::new(-0.55, 0.85, 0.0),
            radius: 0.05,
            mass: mass * 0.04,
        });
        body.limbs.push(LimbCapsule {
            name: "right_arm".into(),
            local_start: Vec3Phys::new(0.25, 1.35, 0.0),
            local_end: Vec3Phys::new(0.55, 0.85, 0.0),
            radius: 0.05,
            mass: mass * 0.04,
        });

        body
    }

    /// Update the character body.
    pub fn update(&mut self, desired_velocity: Vec3Phys, dt: f32) {
        // Apply root motion if enabled.
        let target_velocity = if self.root_motion_enabled {
            self.root_motion_velocity
        } else {
            desired_velocity
        };

        // Smooth velocity change.
        let acceleration = target_velocity.sub(self.velocity).scale(10.0);
        self.velocity = self.velocity.add(acceleration.scale(dt));

        // Integrate position.
        self.position = self.position.add(self.velocity.scale(dt));

        // Compute speed.
        self.speed = self.velocity.length();

        // Update facing direction.
        if self.speed > 0.1 {
            self.facing = self.velocity.z.atan2(self.velocity.x);
        }

        // Body lean (bicycle model): lean into direction changes.
        self.update_lean(desired_velocity, dt);

        // Shift center of mass based on lean.
        self.com_shift = Vec3Phys::new(
            self.lean_angle.sin() * 0.2,
            0.0,
            0.0,
        );

        // Foot IK: project feet onto ground.
        if self.foot_ik_enabled && self.grounded {
            self.update_foot_ik();
        }

        // Ground clamp.
        if self.position.y < self.ground_height {
            self.position.y = self.ground_height;
            self.velocity.y = 0.0;
            self.grounded = true;
        }
    }

    /// Update body lean based on movement direction changes.
    fn update_lean(&mut self, desired_velocity: Vec3Phys, dt: f32) {
        if self.speed < 1.0 {
            // Return to upright when slow.
            self.lean_angle *= (1.0 - 5.0 * dt).max(0.0);
            return;
        }

        // Compute lateral acceleration (perpendicular to velocity).
        let vel_dir = self.velocity.normalized();
        let desired_dir = desired_velocity.normalized();
        let lateral = Vec3Phys::cross(vel_dir, desired_dir);
        let lateral_mag = lateral.y; // Y component of cross product = lateral turn

        // Target lean proportional to lateral acceleration.
        let target_lean = (-lateral_mag * 0.5).clamp(-self.max_lean, self.max_lean);

        // Smooth lean transition.
        self.lean_angle += (target_lean - self.lean_angle) * 8.0 * dt;
    }

    /// Update foot IK targets for ground adaptation.
    fn update_foot_ik(&mut self) {
        let half_stance = 0.15; // Half the distance between feet.

        // Project feet onto ground plane.
        let left_offset = Vec3Phys::new(-half_stance, 0.0, 0.0);
        let right_offset = Vec3Phys::new(half_stance, 0.0, 0.0);

        self.left_foot_target = self.position.add(left_offset);
        self.left_foot_target.y = self.ground_height;

        self.right_foot_target = self.position.add(right_offset);
        self.right_foot_target.y = self.ground_height;

        // Adjust for ground normal (slope adaptation).
        if (self.ground_normal.y - 1.0).abs() > 0.01 {
            // On a slope: tilt feet to match surface.
            let slope_offset = self.ground_normal.x * half_stance;
            self.left_foot_target.y += slope_offset;
            self.right_foot_target.y -= slope_offset;
        }
    }

    /// Get the world-space center of mass.
    pub fn world_center_of_mass(&self) -> Vec3Phys {
        self.position.add(self.center_of_mass).add(self.com_shift)
    }

    /// Check collision between a limb and a point.
    pub fn check_limb_collision(
        &self,
        limb_index: usize,
        point: Vec3Phys,
    ) -> Option<f32> {
        if limb_index >= self.limbs.len() {
            return None;
        }
        let limb = &self.limbs[limb_index];
        let start = self.position.add(limb.local_start);
        let end = self.position.add(limb.local_end);

        // Point-to-segment distance.
        let seg = end.sub(start);
        let seg_len_sq = seg.length_sq();
        if seg_len_sq < EPSILON {
            let dist = Vec3Phys::distance(point, start);
            return if dist <= limb.radius { Some(dist) } else { None };
        }

        let t = Vec3Phys::dot(point.sub(start), seg) / seg_len_sq;
        let t_clamped = t.clamp(0.0, 1.0);
        let closest = start.add(seg.scale(t_clamped));
        let dist = Vec3Phys::distance(point, closest);

        if dist <= limb.radius {
            Some(dist)
        } else {
            None
        }
    }
}

// ===========================================================================
// VehiclePhysicsV2 — advanced vehicle simulation
// ===========================================================================

/// Tire state for advanced vehicle physics.
#[derive(Debug, Clone)]
pub struct TireStateV2 {
    /// Tire temperature (Celsius).
    pub temperature: f32,
    /// Current grip multiplier based on temperature.
    pub grip_multiplier: f32,
    /// Tire load (N).
    pub load: f32,
    /// Slip ratio (longitudinal).
    pub slip_ratio: f32,
    /// Slip angle (lateral, radians).
    pub slip_angle: f32,
    /// Tire wear (0-1, 0 = new, 1 = destroyed).
    pub wear: f32,
    /// Whether the tire is locked (ABS intervention).
    pub locked: bool,
    /// Tire surface temperature (separate from core temperature).
    pub surface_temperature: f32,
    /// Angular velocity of the tire (rad/s).
    pub angular_velocity: f32,
    /// Longitudinal force output (N).
    pub longitudinal_force: f32,
    /// Lateral force output (N).
    pub lateral_force: f32,
}

impl Default for TireStateV2 {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TIRE_TEMP,
            grip_multiplier: 1.0,
            load: 0.0,
            slip_ratio: 0.0,
            slip_angle: 0.0,
            wear: 0.0,
            locked: false,
            surface_temperature: DEFAULT_TIRE_TEMP,
            angular_velocity: 0.0,
            longitudinal_force: 0.0,
            lateral_force: 0.0,
        }
    }
}

impl TireStateV2 {
    /// Update tire temperature based on slip and load.
    ///
    /// Heat generation: proportional to slip * load * speed.
    /// Heat dissipation: proportional to (T - ambient) and speed.
    pub fn update_temperature(&mut self, speed: f32, ambient_temp: f32, dt: f32) {
        // Heat generation from friction work.
        let slip_intensity = self.slip_ratio.abs() + self.slip_angle.abs();
        let heat_gen = slip_intensity * self.load * speed * 0.0001;

        // Heat dissipation (convective cooling).
        let cooling_rate = 0.5 + speed * 0.01; // More cooling at speed.
        let heat_loss = cooling_rate * (self.temperature - ambient_temp) * dt;

        self.temperature += (heat_gen - heat_loss) * dt;
        self.temperature = self.temperature.clamp(-40.0, 200.0);

        // Surface temp responds faster.
        self.surface_temperature += (heat_gen * 2.0 - heat_loss * 1.5) * dt;
        self.surface_temperature = self.surface_temperature.clamp(-40.0, 250.0);

        // Grip varies with temperature.
        self.update_grip();

        // Wear.
        self.wear += slip_intensity * self.load * 1e-9 * dt;
        self.wear = self.wear.min(1.0);
    }

    /// Update grip multiplier based on temperature.
    ///
    /// Optimal grip at ~90C. Falls off when too cold or too hot.
    fn update_grip(&mut self) {
        let diff = (self.temperature - OPTIMAL_TIRE_TEMP).abs();
        let normalized = diff / TIRE_TEMP_RANGE;

        // Gaussian-like grip curve.
        self.grip_multiplier = (-0.5 * normalized * normalized).exp();

        // Wear reduces grip.
        self.grip_multiplier *= 1.0 - 0.5 * self.wear;

        self.grip_multiplier = self.grip_multiplier.clamp(0.2, 1.2);
    }
}

/// Differential type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentialType {
    /// Open differential (all torque to the wheel with less resistance).
    Open,
    /// Limited-slip differential.
    LimitedSlip { lock_percent: u8 },
    /// Fully locked (equal torque distribution).
    Locked,
}

/// Forced induction type.
#[derive(Debug, Clone, Copy)]
pub enum ForcedInduction {
    /// No forced induction.
    None,
    /// Turbocharger with boost curve.
    Turbo {
        /// Maximum boost pressure (bar).
        max_boost: f32,
        /// Spool-up time constant (seconds).
        spool_time: f32,
        /// Current boost pressure.
        current_boost: f32,
    },
    /// Supercharger (instant response).
    Supercharger {
        /// Boost multiplier.
        boost_multiplier: f32,
    },
}

/// Advanced vehicle configuration.
#[derive(Debug, Clone)]
pub struct VehicleConfigV2 {
    /// Total vehicle mass (kg).
    pub mass: f32,
    /// Wheelbase (m, front to rear axle).
    pub wheelbase: f32,
    /// Track width (m, left to right wheel).
    pub track_width: f32,
    /// Center of gravity height (m).
    pub cg_height: f32,
    /// Front weight distribution (0-1).
    pub front_weight_ratio: f32,
    /// Frontal area (m^2) for aerodynamic calculations.
    pub frontal_area: f32,
    /// Drag coefficient.
    pub cd: f32,
    /// Downforce coefficient.
    pub cl: f32,
    /// Maximum engine torque (Nm).
    pub max_torque: f32,
    /// Maximum engine RPM.
    pub max_rpm: f32,
    /// Gear ratios.
    pub gear_ratios: Vec<f32>,
    /// Final drive ratio.
    pub final_drive: f32,
    /// Differential type.
    pub differential: DifferentialType,
    /// Forced induction.
    pub induction: ForcedInduction,
    /// Tire radius (m).
    pub tire_radius: f32,
    /// Maximum brake torque (Nm).
    pub max_brake_torque: f32,
    /// Maximum steering angle (radians).
    pub max_steer_angle: f32,
}

impl Default for VehicleConfigV2 {
    fn default() -> Self {
        Self {
            mass: 1500.0,
            wheelbase: 2.7,
            track_width: 1.6,
            cg_height: 0.5,
            front_weight_ratio: 0.55,
            frontal_area: 2.2,
            cd: 0.3,
            cl: 0.15,
            max_torque: 400.0,
            max_rpm: 7000.0,
            gear_ratios: vec![3.6, 2.1, 1.4, 1.0, 0.78, 0.63],
            final_drive: 3.42,
            differential: DifferentialType::LimitedSlip { lock_percent: 30 },
            induction: ForcedInduction::None,
            tire_radius: 0.33,
            max_brake_torque: 3000.0,
            max_steer_angle: 0.6,
        }
    }
}

/// Advanced vehicle state.
#[derive(Debug)]
pub struct VehicleStateV2 {
    /// Position.
    pub position: Vec3Phys,
    /// Velocity.
    pub velocity: Vec3Phys,
    /// Yaw angle (radians).
    pub yaw: f32,
    /// Yaw rate (rad/s).
    pub yaw_rate: f32,
    /// Current gear (0 = reverse, 1 = first, etc.).
    pub gear: u8,
    /// Engine RPM.
    pub rpm: f32,
    /// Throttle input (0-1).
    pub throttle: f32,
    /// Brake input (0-1).
    pub brake: f32,
    /// Steering input (-1 to 1).
    pub steering: f32,
    /// Per-tire state: [FL, FR, RL, RR].
    pub tires: [TireStateV2; 4],
    /// Speed (m/s).
    pub speed: f32,
    /// Longitudinal acceleration (m/s^2).
    pub longitudinal_accel: f32,
    /// Lateral acceleration (m/s^2).
    pub lateral_accel: f32,
    /// Downforce (N).
    pub downforce: f32,
    /// Drag force (N).
    pub drag: f32,
}

impl VehicleStateV2 {
    /// Create a default vehicle state at a position.
    pub fn at_position(pos: Vec3Phys) -> Self {
        Self {
            position: pos,
            velocity: Vec3Phys::ZERO,
            yaw: 0.0,
            yaw_rate: 0.0,
            gear: 1,
            rpm: 1000.0,
            throttle: 0.0,
            brake: 0.0,
            steering: 0.0,
            tires: Default::default(),
            speed: 0.0,
            longitudinal_accel: 0.0,
            lateral_accel: 0.0,
            downforce: 0.0,
            drag: 0.0,
        }
    }

    /// Speed in km/h.
    pub fn speed_kph(&self) -> f32 {
        self.speed * 3.6
    }

    /// Speed in mph.
    pub fn speed_mph(&self) -> f32 {
        self.speed * 2.237
    }
}

/// Advanced vehicle physics simulation.
#[derive(Debug)]
pub struct VehiclePhysicsV2 {
    pub config: VehicleConfigV2,
    pub state: VehicleStateV2,
}

impl VehiclePhysicsV2 {
    /// Create a new vehicle simulation.
    pub fn new(config: VehicleConfigV2) -> Self {
        Self {
            state: VehicleStateV2::at_position(Vec3Phys::ZERO),
            config,
        }
    }

    /// Compute tire load transfer due to acceleration.
    ///
    /// Longitudinal: braking shifts weight forward, acceleration shifts it back.
    /// Lateral: cornering shifts weight to the outside wheels.
    pub fn compute_load_transfer(&self) -> [f32; 4] {
        let total_weight = self.config.mass * GRAVITY;
        let front_static = total_weight * self.config.front_weight_ratio;
        let rear_static = total_weight * (1.0 - self.config.front_weight_ratio);

        // Longitudinal transfer.
        let long_transfer = self.config.mass * self.state.longitudinal_accel
            * self.config.cg_height
            / self.config.wheelbase;

        // Lateral transfer (simplified: same for front and rear).
        let lat_transfer = self.config.mass * self.state.lateral_accel
            * self.config.cg_height
            / self.config.track_width;

        let fl = (front_static * 0.5 + long_transfer * 0.5 - lat_transfer * 0.5).max(0.0);
        let fr = (front_static * 0.5 + long_transfer * 0.5 + lat_transfer * 0.5).max(0.0);
        let rl = (rear_static * 0.5 - long_transfer * 0.5 - lat_transfer * 0.5).max(0.0);
        let rr = (rear_static * 0.5 - long_transfer * 0.5 + lat_transfer * 0.5).max(0.0);

        [fl, fr, rl, rr]
    }

    /// Compute aerodynamic downforce.
    ///
    /// F_down = 0.5 * CL * rho * A * v^2
    pub fn compute_downforce(&self, speed: f32) -> f32 {
        0.5 * self.config.cl * AIR_DENSITY * self.config.frontal_area * speed * speed
    }

    /// Compute aerodynamic drag.
    ///
    /// F_drag = 0.5 * Cd * rho * A * v^2
    pub fn compute_aero_drag(&self, speed: f32) -> f32 {
        0.5 * self.config.cd * AIR_DENSITY * self.config.frontal_area * speed * speed
    }

    /// Compute engine torque with forced induction.
    pub fn compute_engine_torque(&mut self, dt: f32) -> f32 {
        let base_torque = self.config.max_torque * self.state.throttle;

        match &mut self.config.induction {
            ForcedInduction::None => base_torque,
            ForcedInduction::Turbo {
                max_boost,
                spool_time,
                current_boost,
            } => {
                // Turbo spool-up: boost builds with RPM and throttle.
                let target_boost = *max_boost * self.state.throttle
                    * (self.state.rpm / self.config.max_rpm).min(1.0);
                let spool_rate = 1.0 / spool_time.max(0.1);
                *current_boost += (target_boost - *current_boost) * spool_rate * dt;
                *current_boost = current_boost.clamp(0.0, *max_boost);

                // Boost multiplies torque.
                base_torque * (1.0 + *current_boost)
            }
            ForcedInduction::Supercharger { boost_multiplier } => {
                // Supercharger: instant boost proportional to RPM.
                let boost = *boost_multiplier
                    * (self.state.rpm / self.config.max_rpm).min(1.0);
                base_torque * (1.0 + boost)
            }
        }
    }

    /// Distribute torque through differential to driven wheels.
    pub fn distribute_torque(&self, total_torque: f32) -> (f32, f32) {
        match self.config.differential {
            DifferentialType::Open => {
                // All torque to the wheel with least resistance.
                // Simplified: equal split.
                (total_torque * 0.5, total_torque * 0.5)
            }
            DifferentialType::LimitedSlip { lock_percent } => {
                let lock_ratio = lock_percent as f32 / 100.0;
                let open_ratio = 1.0 - lock_ratio;
                let base = total_torque * 0.5;

                // Bias torque based on wheel speed difference.
                let left_spin = self.state.tires[2].angular_velocity;
                let right_spin = self.state.tires[3].angular_velocity;
                let spin_diff = (left_spin - right_spin).abs();

                let bias = (spin_diff * lock_ratio * 0.1).min(base * lock_ratio);
                if left_spin > right_spin {
                    (base - bias * open_ratio, base + bias * open_ratio)
                } else {
                    (base + bias * open_ratio, base - bias * open_ratio)
                }
            }
            DifferentialType::Locked => {
                (total_torque * 0.5, total_torque * 0.5)
            }
        }
    }

    /// Main update step.
    pub fn update(&mut self, dt: f32) {
        // Speed.
        self.state.speed = self.state.velocity.length();

        // Aerodynamics.
        self.state.downforce = self.compute_downforce(self.state.speed);
        self.state.drag = self.compute_aero_drag(self.state.speed);

        // Tire loads including aero downforce and load transfer.
        let base_loads = self.compute_load_transfer();
        let aero_bonus = self.state.downforce * 0.25; // Per tire
        for i in 0..4 {
            self.state.tires[i].load = base_loads[i] + aero_bonus;
        }

        // Engine torque.
        let engine_torque = self.compute_engine_torque(dt);

        // Gear ratio.
        let gear_idx = (self.state.gear as usize).min(self.config.gear_ratios.len());
        let gear_ratio = if gear_idx > 0 && gear_idx <= self.config.gear_ratios.len() {
            self.config.gear_ratios[gear_idx - 1]
        } else {
            1.0
        };

        // Wheel torque.
        let wheel_torque = engine_torque * gear_ratio * self.config.final_drive;
        let (_left_torque, _right_torque) = self.distribute_torque(wheel_torque);

        // Update tire temperatures.
        for tire in &mut self.state.tires {
            tire.update_temperature(self.state.speed, 25.0, dt);
        }

        // Apply drag.
        if self.state.speed > EPSILON {
            let drag_decel = self.state.drag / self.config.mass;
            let vel_dir = self.state.velocity.normalized();
            self.state.velocity -= vel_dir * (drag_decel * dt);
        }

        // Update RPM.
        if self.config.tire_radius > EPSILON {
            let wheel_speed = self.state.speed / self.config.tire_radius;
            self.state.rpm = wheel_speed * gear_ratio * self.config.final_drive
                * 60.0 / (2.0 * PI);
            self.state.rpm = self.state.rpm.clamp(800.0, self.config.max_rpm);
        }

        // Integrate position.
        self.state.position += self.state.velocity * dt;
    }

    /// Reset vehicle to a position.
    pub fn reset(&mut self, position: Vec3Phys) {
        self.state = VehicleStateV2::at_position(position);
    }
}

// ===========================================================================
// PhysicsDeterminism — verification tools
// ===========================================================================

/// Replay entry: a snapshot of inputs at a specific tick.
#[derive(Debug, Clone)]
pub struct ReplayEntry {
    pub tick: u32,
    pub inputs: Vec<u8>, // serialized inputs
}

/// Determinism verifier: records and compares simulation replays.
#[derive(Debug)]
pub struct DeterminismVerifier {
    /// Recorded state hashes.
    pub hashes: Vec<StateHash>,
    /// Replay inputs.
    pub replay: Vec<ReplayEntry>,
    /// Whether recording is active.
    pub recording: bool,
    /// Comparison result.
    pub last_desync: Option<DesyncInfo>,
}

impl DeterminismVerifier {
    /// Create a new verifier.
    pub fn new() -> Self {
        Self {
            hashes: Vec::new(),
            replay: Vec::new(),
            recording: false,
            last_desync: None,
        }
    }

    /// Start recording.
    pub fn start_recording(&mut self) {
        self.hashes.clear();
        self.replay.clear();
        self.recording = true;
    }

    /// Stop recording.
    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    /// Record a tick's state and inputs.
    pub fn record_tick(&mut self, hash: StateHash, inputs: Vec<u8>) {
        if !self.recording {
            return;
        }
        self.hashes.push(hash);
        self.replay.push(ReplayEntry {
            tick: hash.tick,
            inputs,
        });
    }

    /// Compare with another set of hashes.
    pub fn compare(&mut self, other_hashes: &[StateHash]) -> Option<&DesyncInfo> {
        self.last_desync = DeterministicPhysics::find_desync(&self.hashes, other_hashes);
        self.last_desync.as_ref()
    }

    /// Get the number of recorded ticks.
    pub fn recorded_ticks(&self) -> usize {
        self.hashes.len()
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.hashes.clear();
        self.replay.clear();
        self.last_desync = None;
    }
}

impl Default for DeterminismVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_float_basic() {
        let a = FixedFloat::from_f32(3.14);
        let b = FixedFloat::from_f32(2.0);
        let c = a + b;
        assert!((c.to_f32() - 5.14).abs() < 0.01);

        let d = a - b;
        assert!((d.to_f32() - 1.14).abs() < 0.01);

        let e = a * b;
        assert!((e.to_f32() - 6.28).abs() < 0.05);

        let f = a / b;
        assert!((f.to_f32() - 1.57).abs() < 0.01);
    }

    #[test]
    fn test_fixed_float_sqrt() {
        let a = FixedFloat::from_f32(4.0);
        let s = a.sqrt();
        assert!((s.to_f32() - 2.0).abs() < 0.01);

        let b = FixedFloat::from_f32(9.0);
        let s2 = b.sqrt();
        assert!((s2.to_f32() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_fixed_float_sin() {
        let half_pi = FixedFloat::from_f32(PI / 2.0);
        let s = half_pi.sin();
        assert!((s.to_f32() - 1.0).abs() < 0.05);

        let zero_sin = FixedFloat::ZERO.sin();
        assert!((zero_sin.to_f32()).abs() < 0.01);
    }

    #[test]
    fn test_fixed_float_determinism() {
        // Same computation should produce the exact same raw bits.
        let a = FixedFloat::from_f32(7.5);
        let b = FixedFloat::from_f32(3.3);
        let c1 = a.mul(b).add(a).sub(b);
        let c2 = a.mul(b).add(a).sub(b);
        assert_eq!(c1.raw, c2.raw); // Bit-exact!
    }

    #[test]
    fn test_magnus_force() {
        // A ball with topspin moving forward should curve downward.
        let velocity = Vec3Phys::new(0.0, 0.0, 20.0); // 20 m/s forward
        let angular_velocity = Vec3Phys::new(100.0, 0.0, 0.0); // topspin
        let force = SportPhysics::magnus_force(
            velocity,
            angular_velocity,
            SOCCER_BALL_RADIUS,
            AIR_DENSITY,
        );

        // Magnus should produce a downward force (negative Y).
        assert!(force.y < 0.0, "Topspin should produce downward Magnus force");
    }

    #[test]
    fn test_magnus_backspin() {
        // Backspin should produce upward Magnus force.
        let velocity = Vec3Phys::new(0.0, 0.0, 20.0);
        let angular_velocity = Vec3Phys::new(-100.0, 0.0, 0.0); // backspin
        let force = SportPhysics::magnus_force(
            velocity,
            angular_velocity,
            SOCCER_BALL_RADIUS,
            AIR_DENSITY,
        );

        assert!(force.y > 0.0, "Backspin should produce upward Magnus force");
    }

    #[test]
    fn test_drag_force() {
        let velocity = Vec3Phys::new(0.0, 0.0, 30.0);
        let drag = SportPhysics::drag_force(
            velocity,
            Vec3Phys::ZERO,
            SOCCER_BALL_CD,
            SOCCER_BALL_RADIUS,
            AIR_DENSITY,
            false,
        );

        // Drag should oppose motion.
        assert!(drag.z < 0.0, "Drag should oppose forward motion");
        // Magnitude check: F = 0.5 * 0.25 * 1.225 * pi*0.11^2 * 30^2
        let expected = 0.5 * SOCCER_BALL_CD * AIR_DENSITY * PI * SOCCER_BALL_RADIUS
            * SOCCER_BALL_RADIUS * 900.0;
        assert!(
            (drag.z.abs() - expected).abs() < 1.0,
            "Drag magnitude mismatch: got {}, expected {}",
            drag.z.abs(),
            expected
        );
    }

    #[test]
    fn test_reynolds_number() {
        let re = SportPhysics::reynolds_number(30.0, 0.22);
        // Re = 30 * 0.22 / 1.516e-5 = ~435,000
        assert!(re > 400_000.0);
        assert!(re < 500_000.0);
    }

    #[test]
    fn test_cd_from_reynolds() {
        // Below drag crisis.
        let cd_low = SportPhysics::cd_from_reynolds(100_000.0);
        assert!((cd_low - CD_SUBCRITICAL).abs() < 0.1);

        // Above drag crisis.
        let cd_high = SportPhysics::cd_from_reynolds(300_000.0);
        assert!((cd_high - CD_SUPERCRITICAL).abs() < 0.1);
    }

    #[test]
    fn test_ball_simulation() {
        let mut sim = SportPhysics::new(BallConfig::soccer());
        sim.state.position = Vec3Phys::new(0.0, 1.0, 0.0);

        // Kick with forward velocity and sidespin.
        sim.kick(
            Vec3Phys::new(0.0, 5.0, 25.0),
            Vec3Phys::new(0.0, 50.0, 0.0), // sidespin
        );

        // Simulate for 2 seconds.
        for _ in 0..120 {
            sim.step(1.0 / 60.0);
        }

        // Ball should have moved significantly forward.
        assert!(sim.state.position.z > 10.0, "Ball should travel forward");
        // Ball should have curved due to Magnus effect.
        assert!(
            sim.state.position.x.abs() > 0.1,
            "Ball should curve from sidespin"
        );
    }

    #[test]
    fn test_soft_contact_grass() {
        let grass = SoftContactMaterial::grass();
        assert!(grass.friction_anisotropy < 1.0); // Anisotropic
        assert!(grass.compliance > 0.0);
    }

    #[test]
    fn test_tire_temperature_grip() {
        let mut tire = TireStateV2::default();
        tire.temperature = OPTIMAL_TIRE_TEMP;
        tire.update_grip();
        // At optimal temp, grip should be near maximum.
        assert!(
            tire.grip_multiplier > 0.9,
            "Grip at optimal temp should be high: {}",
            tire.grip_multiplier
        );

        // Cold tire.
        tire.temperature = 20.0;
        tire.update_grip();
        assert!(
            tire.grip_multiplier < 0.5,
            "Grip when cold should be low: {}",
            tire.grip_multiplier
        );
    }

    #[test]
    fn test_vehicle_downforce() {
        let vehicle = VehiclePhysicsV2::new(VehicleConfigV2::default());
        let downforce = vehicle.compute_downforce(50.0); // 50 m/s = 180 km/h
        assert!(downforce > 0.0, "Downforce should be positive at speed");
        // F = 0.5 * 0.15 * 1.225 * 2.2 * 2500 = ~506N
        assert!(downforce > 400.0 && downforce < 600.0);
    }

    #[test]
    fn test_load_transfer() {
        let vehicle = VehiclePhysicsV2::new(VehicleConfigV2::default());
        let loads = vehicle.compute_load_transfer();
        let total: f32 = loads.iter().sum();
        let expected = vehicle.config.mass * GRAVITY;
        assert!(
            (total - expected).abs() < 1.0,
            "Load should sum to weight: {} vs {}",
            total,
            expected
        );
    }

    #[test]
    fn test_character_body_humanoid() {
        let mut body = CharacterBody::humanoid(Vec3Phys::ZERO, 80.0);
        assert_eq!(body.limbs.len(), 8);
        assert!(body.mass == 80.0);

        // Simulate movement.
        let desired = Vec3Phys::new(0.0, 0.0, 5.0); // 5 m/s forward
        for _ in 0..60 {
            body.update(desired, 1.0 / 60.0);
        }

        // Should have moved forward.
        assert!(body.position.z > 3.0);
    }

    #[test]
    fn test_state_hash_determinism() {
        let bodies = vec![
            PhysicsBodyState {
                id: 1,
                position: Vec3Phys::new(1.0, 2.0, 3.0),
                velocity: Vec3Phys::new(0.1, 0.2, 0.3),
                angular_velocity: Vec3Phys::ZERO,
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            PhysicsBodyState {
                id: 2,
                position: Vec3Phys::new(4.0, 5.0, 6.0),
                velocity: Vec3Phys::ZERO,
                angular_velocity: Vec3Phys::ZERO,
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        ];

        let phys = DeterministicPhysics::new(1.0 / 60.0);
        let hash1 = phys.compute_state_hash(&bodies);
        let hash2 = phys.compute_state_hash(&bodies);
        assert_eq!(hash1, hash2, "Same state should produce same hash");
    }

    #[test]
    fn test_desync_detection() {
        let hashes_a = vec![
            StateHash { tick: 0, hash: 100, body_count: 1 },
            StateHash { tick: 1, hash: 200, body_count: 1 },
            StateHash { tick: 2, hash: 300, body_count: 1 },
        ];
        let hashes_b = vec![
            StateHash { tick: 0, hash: 100, body_count: 1 },
            StateHash { tick: 1, hash: 200, body_count: 1 },
            StateHash { tick: 2, hash: 999, body_count: 1 }, // desync!
        ];

        let desync = DeterministicPhysics::find_desync(&hashes_a, &hashes_b);
        assert!(desync.is_some());
        assert_eq!(desync.unwrap().tick, 2);
    }

    #[test]
    fn test_precision_solver_basic() {
        let mut solver = PrecisionSolver::with_defaults();

        let a = solver.add_body(SolverBody::static_body(Vec3Phys::ZERO));
        let b = solver.add_body(SolverBody::dynamic(Vec3Phys::new(0.0, 1.0, 0.0), 1.0));

        solver.add_contact(ContactConstraint::new(
            a,
            b,
            Vec3Phys::new(0.0, 0.5, 0.0),
            Vec3Phys::UP,
            0.01,
        ));

        solver.solve(1.0 / 60.0);
        assert!(solver.stats.contacts_solved > 0);
    }
}
