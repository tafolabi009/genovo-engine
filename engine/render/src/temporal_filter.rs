// engine/render/src/temporal_filter.rs
//
// Generic temporal filtering framework for the Genovo renderer.
//
// Provides accumulation buffer management, motion-vector reprojection,
// neighborhood clamping (AABB and variance clip), velocity rejection,
// sub-pixel jitter patterns, configurable blend factor, and automatic
// reset on camera cuts.
//
// # Architecture
//
// `TemporalFilter` is the main entry point. It manages a pair of
// accumulation buffers (ping-pong), stores per-pixel motion vectors,
// and exposes a `resolve` method that produces the temporally filtered
// output for the current frame.
//
// The filter supports arbitrary data (color, AO, GI, shadow) via
// generic `TemporalSample` trait.

use std::collections::VecDeque;
use std::fmt;

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

/// 2D vector for screen-space operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }
}

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self::new(
            self.x.clamp(lo.x, hi.x),
            self.y.clamp(lo.y, hi.y),
            self.z.clamp(lo.z, hi.z),
        )
    }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )
    }
}

/// 4x4 matrix for view-projection operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        cols: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// Transform a 3D point by this 4x4 matrix (assumes w=1, performs perspective divide).
    #[inline]
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let x = self.cols[0][0] * p.x + self.cols[1][0] * p.y + self.cols[2][0] * p.z + self.cols[3][0];
        let y = self.cols[0][1] * p.x + self.cols[1][1] * p.y + self.cols[2][1] * p.z + self.cols[3][1];
        let z = self.cols[0][2] * p.x + self.cols[1][2] * p.y + self.cols[2][2] * p.z + self.cols[3][2];
        let w = self.cols[0][3] * p.x + self.cols[1][3] * p.y + self.cols[2][3] * p.z + self.cols[3][3];
        let inv_w = if w.abs() > 1e-7 { 1.0 / w } else { 1.0 };
        Vec3::new(x * inv_w, y * inv_w, z * inv_w)
    }

    /// Multiply two matrices.
    pub fn mul(&self, rhs: &Self) -> Self {
        let mut result = [[0.0f32; 4]; 4];
        for c in 0..4 {
            for r in 0..4 {
                result[c][r] = self.cols[0][r] * rhs.cols[c][0]
                    + self.cols[1][r] * rhs.cols[c][1]
                    + self.cols[2][r] * rhs.cols[c][2]
                    + self.cols[3][r] * rhs.cols[c][3];
            }
        }
        Self { cols: result }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default blend factor for temporal accumulation (higher = more history).
pub const DEFAULT_BLEND_FACTOR: f32 = 0.95;

/// Minimum blend factor allowed.
pub const MIN_BLEND_FACTOR: f32 = 0.0;

/// Maximum blend factor allowed.
pub const MAX_BLEND_FACTOR: f32 = 0.99;

/// Velocity magnitude threshold above which the pixel is rejected.
pub const DEFAULT_VELOCITY_REJECTION_THRESHOLD: f32 = 40.0;

/// Standard deviation multiplier for variance clipping (gamma).
pub const DEFAULT_VARIANCE_CLIP_GAMMA: f32 = 1.0;

/// Number of frames before the filter is considered converged.
pub const CONVERGENCE_FRAME_COUNT: u32 = 16;

/// Maximum number of jitter samples in a jitter sequence.
pub const MAX_JITTER_SAMPLES: usize = 64;

/// Camera cut detection threshold (distance in world space).
pub const CAMERA_CUT_DISTANCE_THRESHOLD: f32 = 50.0;

/// Camera cut detection threshold (rotation in radians).
pub const CAMERA_CUT_ROTATION_THRESHOLD: f32 = 1.0;

// ---------------------------------------------------------------------------
// Jitter patterns
// ---------------------------------------------------------------------------

/// Sub-pixel jitter pattern used for temporal anti-aliasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JitterPattern {
    /// No jitter (disabled).
    None,
    /// 2-sample pattern (basic).
    Halton2,
    /// 4-sample Halton sequence.
    Halton4,
    /// 8-sample Halton sequence (recommended for TAA).
    Halton8,
    /// 16-sample Halton sequence (high quality).
    Halton16,
    /// R2 quasi-random sequence (Martin Roberts).
    R2Sequence,
    /// Blue noise jitter from a precomputed table.
    BlueNoise,
    /// Custom user-provided jitter offsets.
    Custom,
}

impl fmt::Display for JitterPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Halton2 => write!(f, "Halton(2)"),
            Self::Halton4 => write!(f, "Halton(4)"),
            Self::Halton8 => write!(f, "Halton(8)"),
            Self::Halton16 => write!(f, "Halton(16)"),
            Self::R2Sequence => write!(f, "R2"),
            Self::BlueNoise => write!(f, "Blue Noise"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Generates Halton sequence value for a given index and base.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0f32;
    let mut denom = 1.0f32;
    while index > 0 {
        denom *= base as f32;
        result += (index % base) as f32 / denom;
        index /= base;
    }
    result
}

/// R2 quasi-random sequence generator.
fn r2_sample(index: u32) -> Vec2 {
    // Plastic constant related constants.
    let g = 1.32471795724474602596;
    let a1 = 1.0 / g;
    let a2 = 1.0 / (g * g);
    let x = (0.5 + a1 * index as f64) % 1.0;
    let y = (0.5 + a2 * index as f64) % 1.0;
    Vec2::new(x as f32 - 0.5, y as f32 - 0.5)
}

/// Jitter sequence generator.
#[derive(Debug, Clone)]
pub struct JitterSequence {
    pattern: JitterPattern,
    samples: Vec<Vec2>,
    current_index: u32,
    total_samples: u32,
}

impl JitterSequence {
    /// Create a new jitter sequence for the given pattern.
    pub fn new(pattern: JitterPattern) -> Self {
        let total = match pattern {
            JitterPattern::None => 1,
            JitterPattern::Halton2 => 2,
            JitterPattern::Halton4 => 4,
            JitterPattern::Halton8 => 8,
            JitterPattern::Halton16 => 16,
            JitterPattern::R2Sequence => 32,
            JitterPattern::BlueNoise => 16,
            JitterPattern::Custom => 1,
        };

        let mut samples = Vec::with_capacity(total as usize);
        match pattern {
            JitterPattern::None => {
                samples.push(Vec2::ZERO);
            }
            JitterPattern::Halton2
            | JitterPattern::Halton4
            | JitterPattern::Halton8
            | JitterPattern::Halton16 => {
                for i in 0..total {
                    let x = halton(i + 1, 2) - 0.5;
                    let y = halton(i + 1, 3) - 0.5;
                    samples.push(Vec2::new(x, y));
                }
            }
            JitterPattern::R2Sequence => {
                for i in 0..total {
                    samples.push(r2_sample(i));
                }
            }
            JitterPattern::BlueNoise => {
                // Precomputed 16-sample blue noise offsets (approximation).
                let blue_noise: [(f32, f32); 16] = [
                    (-0.375, 0.125), (0.125, -0.375), (0.375, 0.375), (-0.125, -0.125),
                    (-0.250, 0.375), (0.375, -0.125), (-0.375, -0.375), (0.125, 0.250),
                    (0.000, 0.000), (-0.125, 0.375), (0.250, -0.250), (-0.375, 0.000),
                    (0.375, 0.125), (-0.250, -0.250), (0.000, 0.375), (0.125, -0.125),
                ];
                for (x, y) in &blue_noise {
                    samples.push(Vec2::new(*x, *y));
                }
            }
            JitterPattern::Custom => {
                samples.push(Vec2::ZERO);
            }
        }

        Self {
            pattern,
            samples,
            current_index: 0,
            total_samples: total,
        }
    }

    /// Create a jitter sequence with custom samples.
    pub fn custom(offsets: &[Vec2]) -> Self {
        let n = offsets.len().min(MAX_JITTER_SAMPLES).max(1);
        let samples = offsets[..n].to_vec();
        Self {
            pattern: JitterPattern::Custom,
            samples,
            current_index: 0,
            total_samples: n as u32,
        }
    }

    /// Get the current jitter offset and advance to the next sample.
    pub fn next_offset(&mut self) -> Vec2 {
        let offset = self.samples[self.current_index as usize];
        self.current_index = (self.current_index + 1) % self.total_samples;
        offset
    }

    /// Get the current jitter offset without advancing.
    pub fn current_offset(&self) -> Vec2 {
        self.samples[self.current_index as usize]
    }

    /// Reset to the beginning of the sequence.
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Get the current frame index within the sequence.
    pub fn frame_index(&self) -> u32 {
        self.current_index
    }

    /// Get the total number of samples in this sequence.
    pub fn sample_count(&self) -> u32 {
        self.total_samples
    }

    /// Get the pattern type.
    pub fn pattern(&self) -> JitterPattern {
        self.pattern
    }

    /// Get all samples in the sequence.
    pub fn all_samples(&self) -> &[Vec2] {
        &self.samples
    }
}

// ---------------------------------------------------------------------------
// Clamping modes
// ---------------------------------------------------------------------------

/// Method used to clamp the history sample to the current neighborhood.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClampingMode {
    /// AABB clamp: clamp the history color to the min/max of the 3x3 neighborhood.
    Aabb,
    /// Variance clip: use mean and standard deviation to compute a tighter AABB.
    VarianceClip,
    /// Combined AABB + variance clip (intersection of both).
    AabbVarianceIntersection,
    /// No clamping (ghosting will be visible on moving objects).
    None,
}

impl fmt::Display for ClampingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aabb => write!(f, "AABB"),
            Self::VarianceClip => write!(f, "Variance Clip"),
            Self::AabbVarianceIntersection => write!(f, "AABB + Variance"),
            Self::None => write!(f, "None"),
        }
    }
}

// ---------------------------------------------------------------------------
// Velocity rejection
// ---------------------------------------------------------------------------

/// How to handle pixels with high velocity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VelocityRejectionMode {
    /// Hard reject: pixels above threshold get zero history weight.
    HardReject,
    /// Soft falloff: blend factor decreases smoothly with velocity magnitude.
    SoftFalloff,
    /// Per-pixel adaptive based on velocity confidence.
    AdaptiveConfidence,
    /// No velocity rejection.
    None,
}

/// Velocity rejection configuration.
#[derive(Debug, Clone, Copy)]
pub struct VelocityRejectionConfig {
    /// Mode for velocity rejection.
    pub mode: VelocityRejectionMode,
    /// Velocity magnitude threshold (pixels per frame).
    pub threshold: f32,
    /// Soft falloff exponent (only for SoftFalloff mode).
    pub falloff_exponent: f32,
    /// Confidence minimum (only for AdaptiveConfidence mode).
    pub confidence_min: f32,
}

impl Default for VelocityRejectionConfig {
    fn default() -> Self {
        Self {
            mode: VelocityRejectionMode::SoftFalloff,
            threshold: DEFAULT_VELOCITY_REJECTION_THRESHOLD,
            falloff_exponent: 2.0,
            confidence_min: 0.1,
        }
    }
}

impl VelocityRejectionConfig {
    /// Compute the blend weight modifier based on the velocity magnitude.
    pub fn compute_weight(&self, velocity_magnitude: f32) -> f32 {
        match self.mode {
            VelocityRejectionMode::HardReject => {
                if velocity_magnitude > self.threshold {
                    0.0
                } else {
                    1.0
                }
            }
            VelocityRejectionMode::SoftFalloff => {
                if velocity_magnitude <= 0.0 {
                    return 1.0;
                }
                let ratio = (velocity_magnitude / self.threshold).min(1.0);
                (1.0 - ratio).powf(self.falloff_exponent)
            }
            VelocityRejectionMode::AdaptiveConfidence => {
                let base = 1.0 - (velocity_magnitude / self.threshold).min(1.0);
                base.max(self.confidence_min)
            }
            VelocityRejectionMode::None => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera cut detection
// ---------------------------------------------------------------------------

/// Camera state for cut detection.
#[derive(Debug, Clone, Copy)]
pub struct CameraState {
    /// Camera world position.
    pub position: Vec3,
    /// Camera forward direction (normalized).
    pub forward: Vec3,
    /// View-projection matrix of the current frame.
    pub view_projection: Mat4,
    /// Inverse view-projection matrix of the current frame.
    pub inv_view_projection: Mat4,
    /// Field of view in radians (vertical).
    pub fov_y: f32,
    /// Near clip plane distance.
    pub near_plane: f32,
    /// Far clip plane distance.
    pub far_plane: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: Vec3::new(0.0, 0.0, -1.0),
            view_projection: Mat4::IDENTITY,
            inv_view_projection: Mat4::IDENTITY,
            fov_y: 1.047, // ~60 degrees
            near_plane: 0.1,
            far_plane: 1000.0,
        }
    }
}

/// Configuration for camera cut detection.
#[derive(Debug, Clone, Copy)]
pub struct CameraCutConfig {
    /// World-space distance threshold for position teleport detection.
    pub distance_threshold: f32,
    /// Rotation threshold in radians.
    pub rotation_threshold: f32,
    /// Enable camera cut detection.
    pub enabled: bool,
    /// Number of frames to hold reset state after a cut.
    pub reset_hold_frames: u32,
}

impl Default for CameraCutConfig {
    fn default() -> Self {
        Self {
            distance_threshold: CAMERA_CUT_DISTANCE_THRESHOLD,
            rotation_threshold: CAMERA_CUT_ROTATION_THRESHOLD,
            enabled: true,
            reset_hold_frames: 2,
        }
    }
}

/// Detects camera cuts (teleports) by comparing consecutive camera states.
#[derive(Debug, Clone)]
pub struct CameraCutDetector {
    config: CameraCutConfig,
    previous_camera: Option<CameraState>,
    frames_since_cut: u32,
    cut_detected: bool,
}

impl CameraCutDetector {
    /// Create a new cut detector with the given configuration.
    pub fn new(config: CameraCutConfig) -> Self {
        Self {
            config,
            previous_camera: None,
            frames_since_cut: u32::MAX,
            cut_detected: false,
        }
    }

    /// Update the detector with the current camera state. Returns `true` if a cut was detected.
    pub fn update(&mut self, camera: &CameraState) -> bool {
        if !self.config.enabled {
            self.previous_camera = Some(*camera);
            self.cut_detected = false;
            return false;
        }

        let cut = if let Some(prev) = &self.previous_camera {
            let dist = camera.position.sub(prev.position).length();
            let dot = camera.forward.dot(prev.forward).clamp(-1.0, 1.0);
            let angle = dot.acos();
            dist > self.config.distance_threshold || angle > self.config.rotation_threshold
        } else {
            // First frame: treat as cut to reset accumulation.
            true
        };

        self.cut_detected = cut;
        if cut {
            self.frames_since_cut = 0;
        } else {
            self.frames_since_cut = self.frames_since_cut.saturating_add(1);
        }

        self.previous_camera = Some(*camera);
        cut
    }

    /// Whether the filter should reset (either cut detected or still within reset hold period).
    pub fn should_reset(&self) -> bool {
        self.frames_since_cut < self.config.reset_hold_frames
    }

    /// Whether a cut was detected on the latest frame.
    pub fn cut_detected(&self) -> bool {
        self.cut_detected
    }

    /// Number of frames since the last camera cut.
    pub fn frames_since_cut(&self) -> u32 {
        self.frames_since_cut
    }

    /// Reset the detector state (e.g. on scene load).
    pub fn reset(&mut self) {
        self.previous_camera = None;
        self.frames_since_cut = u32::MAX;
        self.cut_detected = false;
    }
}

// ---------------------------------------------------------------------------
// Neighborhood clamping
// ---------------------------------------------------------------------------

/// AABB in color space for neighborhood clamping.
#[derive(Debug, Clone, Copy)]
pub struct ColorAabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl ColorAabb {
    /// Create an AABB that encloses nothing (ready for expansion).
    pub fn empty() -> Self {
        Self {
            min: Vec3::new(f32::MAX, f32::MAX, f32::MAX),
            max: Vec3::new(f32::MIN, f32::MIN, f32::MIN),
        }
    }

    /// Expand this AABB to include the given sample.
    #[inline]
    pub fn expand(&mut self, sample: Vec3) {
        self.min = self.min.min(sample);
        self.max = self.max.max(sample);
    }

    /// Clamp a color to lie within this AABB.
    #[inline]
    pub fn clamp_color(&self, color: Vec3) -> Vec3 {
        color.clamp(self.min, self.max)
    }

    /// Test if a color is inside this AABB.
    #[inline]
    pub fn contains(&self, color: Vec3) -> bool {
        color.x >= self.min.x
            && color.x <= self.max.x
            && color.y >= self.min.y
            && color.y <= self.max.y
            && color.z >= self.min.z
            && color.z <= self.max.z
    }

    /// Intersect two AABBs.
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }
}

/// Variance-based color clip region.
#[derive(Debug, Clone, Copy)]
pub struct VarianceClipRegion {
    /// Mean of the neighborhood.
    pub mean: Vec3,
    /// Standard deviation of the neighborhood.
    pub std_dev: Vec3,
    /// Gamma multiplier for the clip region size.
    pub gamma: f32,
}

impl VarianceClipRegion {
    /// Compute the variance clip region from a set of neighborhood samples.
    pub fn compute(samples: &[Vec3], gamma: f32) -> Self {
        let n = samples.len() as f32;
        if n < 1.0 {
            return Self {
                mean: Vec3::ZERO,
                std_dev: Vec3::ZERO,
                gamma,
            };
        }

        // Compute mean.
        let mut sum = Vec3::ZERO;
        for s in samples {
            sum = sum.add(*s);
        }
        let mean = sum.scale(1.0 / n);

        // Compute variance.
        let mut var = Vec3::ZERO;
        for s in samples {
            let d = s.sub(mean);
            var = var.add(Vec3::new(d.x * d.x, d.y * d.y, d.z * d.z));
        }
        let variance = var.scale(1.0 / n);
        let std_dev = Vec3::new(variance.x.sqrt(), variance.y.sqrt(), variance.z.sqrt());

        Self {
            mean,
            std_dev,
            gamma,
        }
    }

    /// Clip a color sample to this variance region.
    pub fn clip(&self, color: Vec3) -> Vec3 {
        let lo = self.mean.sub(self.std_dev.scale(self.gamma));
        let hi = self.mean.add(self.std_dev.scale(self.gamma));
        color.clamp(lo, hi)
    }

    /// Convert to an AABB.
    pub fn to_aabb(&self) -> ColorAabb {
        ColorAabb {
            min: self.mean.sub(self.std_dev.scale(self.gamma)),
            max: self.mean.add(self.std_dev.scale(self.gamma)),
        }
    }
}

/// Neighborhood sampling pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborhoodPattern {
    /// 3x3 grid (9 samples).
    Cross3x3,
    /// Plus pattern (5 samples: center + 4 neighbors).
    Plus,
    /// Diamond pattern (5 samples, 45-degree rotated).
    Diamond,
    /// Wide 5x5 with sparse sampling (13 samples).
    Wide5x5,
}

/// Stores the result of a neighborhood analysis.
#[derive(Debug, Clone)]
pub struct NeighborhoodResult {
    /// AABB encompassing all neighborhood samples.
    pub aabb: ColorAabb,
    /// Variance clip region computed from samples.
    pub variance_clip: VarianceClipRegion,
    /// All samples collected.
    pub samples: Vec<Vec3>,
    /// Number of valid samples.
    pub sample_count: usize,
}

impl NeighborhoodResult {
    /// Clamp a color using the configured mode.
    pub fn clamp_color(&self, color: Vec3, mode: ClampingMode) -> Vec3 {
        match mode {
            ClampingMode::Aabb => self.aabb.clamp_color(color),
            ClampingMode::VarianceClip => self.variance_clip.clip(color),
            ClampingMode::AabbVarianceIntersection => {
                let var_aabb = self.variance_clip.to_aabb();
                let combined = self.aabb.intersection(&var_aabb);
                combined.clamp_color(color)
            }
            ClampingMode::None => color,
        }
    }
}

// ---------------------------------------------------------------------------
// Motion vector buffer
// ---------------------------------------------------------------------------

/// Per-pixel motion vector for temporal reprojection.
#[derive(Debug, Clone, Copy)]
pub struct MotionVector {
    /// Screen-space motion (in UV coordinates, [0..1] range).
    pub uv_delta: Vec2,
    /// Confidence of this motion vector (0 = unreliable, 1 = high confidence).
    pub confidence: f32,
    /// Whether this pixel had a disocclusion (newly visible).
    pub disoccluded: bool,
}

impl Default for MotionVector {
    fn default() -> Self {
        Self {
            uv_delta: Vec2::ZERO,
            confidence: 1.0,
            disoccluded: false,
        }
    }
}

/// Buffer of per-pixel motion vectors.
#[derive(Debug, Clone)]
pub struct MotionVectorBuffer {
    width: u32,
    height: u32,
    vectors: Vec<MotionVector>,
}

impl MotionVectorBuffer {
    /// Create a new motion vector buffer.
    pub fn new(width: u32, height: u32) -> Self {
        let len = (width * height) as usize;
        Self {
            width,
            height,
            vectors: vec![MotionVector::default(); len],
        }
    }

    /// Resize the buffer (clears all data).
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.vectors.resize((width * height) as usize, MotionVector::default());
        self.clear();
    }

    /// Clear all motion vectors to zero.
    pub fn clear(&mut self) {
        for v in &mut self.vectors {
            *v = MotionVector::default();
        }
    }

    /// Set the motion vector at a given pixel.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, mv: MotionVector) {
        if x < self.width && y < self.height {
            self.vectors[(y * self.width + x) as usize] = mv;
        }
    }

    /// Get the motion vector at a given pixel.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> MotionVector {
        if x < self.width && y < self.height {
            self.vectors[(y * self.width + x) as usize]
        } else {
            MotionVector::default()
        }
    }

    /// Sample the motion vector with bilinear interpolation at UV coordinates.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> MotionVector {
        let fx = u * (self.width as f32 - 1.0);
        let fy = v * (self.height as f32 - 1.0);

        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let fx_frac = fx - fx.floor();
        let fy_frac = fy - fy.floor();

        let tl = self.get(x0, y0);
        let tr = self.get(x1, y0);
        let bl = self.get(x0, y1);
        let br = self.get(x1, y1);

        let top_uv = tl.uv_delta.scale(1.0 - fx_frac).add(tr.uv_delta.scale(fx_frac));
        let bot_uv = bl.uv_delta.scale(1.0 - fx_frac).add(br.uv_delta.scale(fx_frac));
        let uv = top_uv.scale(1.0 - fy_frac).add(bot_uv.scale(fy_frac));

        let conf = tl.confidence * (1.0 - fx_frac) * (1.0 - fy_frac)
            + tr.confidence * fx_frac * (1.0 - fy_frac)
            + bl.confidence * (1.0 - fx_frac) * fy_frac
            + br.confidence * fx_frac * fy_frac;

        MotionVector {
            uv_delta: uv,
            confidence: conf,
            disoccluded: tl.disoccluded || tr.disoccluded || bl.disoccluded || br.disoccluded,
        }
    }

    /// Get the buffer dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the raw motion vector slice.
    pub fn as_slice(&self) -> &[MotionVector] {
        &self.vectors
    }

    /// Get a mutable reference to the raw motion vector slice.
    pub fn as_mut_slice(&mut self) -> &mut [MotionVector] {
        &mut self.vectors
    }
}

// ---------------------------------------------------------------------------
// Accumulation buffer
// ---------------------------------------------------------------------------

/// A single pixel in the accumulation buffer.
#[derive(Debug, Clone, Copy)]
pub struct AccumulationPixel {
    /// Accumulated color (RGB).
    pub color: Vec3,
    /// Depth at this pixel for depth comparison.
    pub depth: f32,
    /// Number of frames this pixel has been accumulating.
    pub frame_count: u16,
    /// Whether this pixel is valid for blending.
    pub valid: bool,
}

impl Default for AccumulationPixel {
    fn default() -> Self {
        Self {
            color: Vec3::ZERO,
            depth: 1.0,
            frame_count: 0,
            valid: false,
        }
    }
}

/// Ping-pong accumulation buffer pair.
#[derive(Debug, Clone)]
pub struct AccumulationBuffer {
    width: u32,
    height: u32,
    /// Current buffer (read).
    current: Vec<AccumulationPixel>,
    /// Previous buffer (history).
    history: Vec<AccumulationPixel>,
    /// Which buffer is "current" (for ping-pong swap).
    current_is_a: bool,
}

impl AccumulationBuffer {
    /// Create a new accumulation buffer pair.
    pub fn new(width: u32, height: u32) -> Self {
        let len = (width * height) as usize;
        Self {
            width,
            height,
            current: vec![AccumulationPixel::default(); len],
            history: vec![AccumulationPixel::default(); len],
            current_is_a: true,
        }
    }

    /// Resize the buffers (clears all data).
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let len = (width * height) as usize;
        self.current = vec![AccumulationPixel::default(); len];
        self.history = vec![AccumulationPixel::default(); len];
    }

    /// Swap current and history for the next frame.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.history);
        self.current_is_a = !self.current_is_a;
    }

    /// Clear all accumulation data.
    pub fn clear(&mut self) {
        for p in &mut self.current {
            *p = AccumulationPixel::default();
        }
        for p in &mut self.history {
            *p = AccumulationPixel::default();
        }
    }

    /// Set a pixel in the current buffer.
    #[inline]
    pub fn set_current(&mut self, x: u32, y: u32, pixel: AccumulationPixel) {
        if x < self.width && y < self.height {
            self.current[(y * self.width + x) as usize] = pixel;
        }
    }

    /// Get a pixel from the current buffer.
    #[inline]
    pub fn get_current(&self, x: u32, y: u32) -> AccumulationPixel {
        if x < self.width && y < self.height {
            self.current[(y * self.width + x) as usize]
        } else {
            AccumulationPixel::default()
        }
    }

    /// Get a pixel from the history buffer.
    #[inline]
    pub fn get_history(&self, x: u32, y: u32) -> AccumulationPixel {
        if x < self.width && y < self.height {
            self.history[(y * self.width + x) as usize]
        } else {
            AccumulationPixel::default()
        }
    }

    /// Sample the history buffer with bilinear interpolation at UV coordinates.
    pub fn sample_history_bilinear(&self, u: f32, v: f32) -> AccumulationPixel {
        let fx = u * (self.width as f32 - 1.0);
        let fy = v * (self.height as f32 - 1.0);

        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(self.width.saturating_sub(1));
        let y1 = (y0 + 1).min(self.height.saturating_sub(1));

        let fx_frac = fx - fx.floor();
        let fy_frac = fy - fy.floor();

        let tl = self.get_history(x0, y0);
        let tr = self.get_history(x1, y0);
        let bl = self.get_history(x0, y1);
        let br = self.get_history(x1, y1);

        let w_tl = (1.0 - fx_frac) * (1.0 - fy_frac);
        let w_tr = fx_frac * (1.0 - fy_frac);
        let w_bl = (1.0 - fx_frac) * fy_frac;
        let w_br = fx_frac * fy_frac;

        let color = tl.color.scale(w_tl)
            .add(tr.color.scale(w_tr))
            .add(bl.color.scale(w_bl))
            .add(br.color.scale(w_br));

        let depth = tl.depth * w_tl + tr.depth * w_tr + bl.depth * w_bl + br.depth * w_br;

        let valid = tl.valid || tr.valid || bl.valid || br.valid;

        let frame_count = (tl.frame_count as f32 * w_tl
            + tr.frame_count as f32 * w_tr
            + bl.frame_count as f32 * w_bl
            + br.frame_count as f32 * w_br) as u16;

        AccumulationPixel {
            color,
            depth,
            frame_count,
            valid,
        }
    }

    /// Get buffer dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get total pixel count.
    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }
}

// ---------------------------------------------------------------------------
// Temporal filter configuration
// ---------------------------------------------------------------------------

/// Complete configuration for the temporal filter.
#[derive(Debug, Clone)]
pub struct TemporalFilterConfig {
    /// Base blend factor (0..1). Higher = more history, smoother but more ghosting.
    pub blend_factor: f32,
    /// Neighborhood clamping mode.
    pub clamping_mode: ClampingMode,
    /// Variance clip gamma (standard deviation multiplier).
    pub variance_gamma: f32,
    /// Jitter pattern for sub-pixel sampling.
    pub jitter_pattern: JitterPattern,
    /// Velocity rejection configuration.
    pub velocity_rejection: VelocityRejectionConfig,
    /// Camera cut detection configuration.
    pub camera_cut: CameraCutConfig,
    /// Neighborhood sampling pattern.
    pub neighborhood_pattern: NeighborhoodPattern,
    /// Enable depth-based rejection (reject history if depth mismatch is large).
    pub depth_rejection_enabled: bool,
    /// Depth rejection threshold (relative depth difference).
    pub depth_rejection_threshold: f32,
    /// Tonemapping before blending (reduces ringing on bright pixels).
    pub tonemap_before_blend: bool,
    /// Enable sharpening of the output.
    pub sharpen_output: bool,
    /// Sharpening strength (0..1).
    pub sharpen_strength: f32,
    /// Responsive anti-aliasing factor (reduces blend for pixels with high luminance difference).
    pub responsive_factor: f32,
}

impl Default for TemporalFilterConfig {
    fn default() -> Self {
        Self {
            blend_factor: DEFAULT_BLEND_FACTOR,
            clamping_mode: ClampingMode::VarianceClip,
            variance_gamma: DEFAULT_VARIANCE_CLIP_GAMMA,
            jitter_pattern: JitterPattern::Halton8,
            velocity_rejection: VelocityRejectionConfig::default(),
            camera_cut: CameraCutConfig::default(),
            neighborhood_pattern: NeighborhoodPattern::Cross3x3,
            depth_rejection_enabled: true,
            depth_rejection_threshold: 0.05,
            tonemap_before_blend: true,
            sharpen_output: false,
            sharpen_strength: 0.2,
            responsive_factor: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics for the temporal filter.
#[derive(Debug, Clone, Copy, Default)]
pub struct TemporalFilterStats {
    /// Number of pixels resolved this frame.
    pub pixels_resolved: u32,
    /// Number of pixels that were reset (no valid history).
    pub pixels_reset: u32,
    /// Number of pixels rejected due to velocity.
    pub pixels_velocity_rejected: u32,
    /// Number of pixels rejected due to depth mismatch.
    pub pixels_depth_rejected: u32,
    /// Number of pixels clamped by neighborhood.
    pub pixels_clamped: u32,
    /// Number of disoccluded pixels.
    pub pixels_disoccluded: u32,
    /// Average blend factor used.
    pub avg_blend_factor: f32,
    /// Whether a camera cut was detected.
    pub camera_cut_detected: bool,
    /// Current jitter frame index.
    pub jitter_frame_index: u32,
    /// Average convergence (average frame count per pixel).
    pub avg_convergence: f32,
}

// ---------------------------------------------------------------------------
// Temporal filter
// ---------------------------------------------------------------------------

/// Main temporal filtering system.
///
/// Manages accumulation buffers, motion vector reprojection, neighborhood
/// clamping, and all related temporal filtering operations.
pub struct TemporalFilter {
    config: TemporalFilterConfig,
    accumulation: AccumulationBuffer,
    motion_vectors: MotionVectorBuffer,
    jitter: JitterSequence,
    cut_detector: CameraCutDetector,
    stats: TemporalFilterStats,
    frame_count: u64,
    width: u32,
    height: u32,
}

impl TemporalFilter {
    /// Create a new temporal filter with the given resolution and configuration.
    pub fn new(width: u32, height: u32, config: TemporalFilterConfig) -> Self {
        let jitter = JitterSequence::new(config.jitter_pattern);
        let cut_detector = CameraCutDetector::new(config.camera_cut);
        Self {
            accumulation: AccumulationBuffer::new(width, height),
            motion_vectors: MotionVectorBuffer::new(width, height),
            jitter,
            cut_detector,
            config,
            stats: TemporalFilterStats::default(),
            frame_count: 0,
            width,
            height,
        }
    }

    /// Resize the filter for a new resolution.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.accumulation.resize(width, height);
        self.motion_vectors.resize(width, height);
        self.jitter.reset();
        self.frame_count = 0;
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: TemporalFilterConfig) {
        if config.jitter_pattern != self.config.jitter_pattern {
            self.jitter = JitterSequence::new(config.jitter_pattern);
        }
        if config.camera_cut.distance_threshold != self.config.camera_cut.distance_threshold
            || config.camera_cut.rotation_threshold != self.config.camera_cut.rotation_threshold
        {
            self.cut_detector = CameraCutDetector::new(config.camera_cut);
        }
        self.config = config;
    }

    /// Get current configuration.
    pub fn config(&self) -> &TemporalFilterConfig {
        &self.config
    }

    /// Get current jitter offset for this frame.
    pub fn current_jitter(&self) -> Vec2 {
        self.jitter.current_offset()
    }

    /// Begin a new frame: advance jitter, detect camera cuts, swap buffers.
    pub fn begin_frame(&mut self, camera: &CameraState) {
        // Detect camera cuts.
        self.cut_detector.update(camera);

        // Swap accumulation buffers.
        self.accumulation.swap();

        // Reset stats.
        self.stats = TemporalFilterStats::default();
        self.stats.jitter_frame_index = self.jitter.frame_index();
        self.stats.camera_cut_detected = self.cut_detector.cut_detected();

        // If camera cut, clear accumulation.
        if self.cut_detector.should_reset() {
            self.accumulation.clear();
        }

        self.frame_count += 1;
    }

    /// Advance jitter to the next sample. Call after begin_frame.
    pub fn advance_jitter(&mut self) -> Vec2 {
        self.jitter.next_offset()
    }

    /// Set motion vectors for this frame.
    pub fn set_motion_vectors(&mut self, vectors: &MotionVectorBuffer) {
        if vectors.dimensions() == (self.width, self.height) {
            self.motion_vectors = vectors.clone();
        }
    }

    /// Get mutable access to the motion vector buffer.
    pub fn motion_vectors_mut(&mut self) -> &mut MotionVectorBuffer {
        &mut self.motion_vectors
    }

    /// Get the motion vector buffer.
    pub fn motion_vectors(&self) -> &MotionVectorBuffer {
        &self.motion_vectors
    }

    /// Tonemap a color for accumulation (Reinhard).
    #[inline]
    fn tonemap(color: Vec3) -> Vec3 {
        Vec3::new(
            color.x / (1.0 + color.x),
            color.y / (1.0 + color.y),
            color.z / (1.0 + color.z),
        )
    }

    /// Inverse tonemap.
    #[inline]
    fn tonemap_inverse(color: Vec3) -> Vec3 {
        Vec3::new(
            color.x / (1.0 - color.x).max(1e-6),
            color.y / (1.0 - color.y).max(1e-6),
            color.z / (1.0 - color.z).max(1e-6),
        )
    }

    /// Luminance of an RGB color (BT.709).
    #[inline]
    fn luminance(color: Vec3) -> f32 {
        0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z
    }

    /// Resolve a single pixel, blending current color with reprojected history.
    pub fn resolve_pixel(
        &mut self,
        x: u32,
        y: u32,
        current_color: Vec3,
        current_depth: f32,
        neighborhood_samples: &[Vec3],
    ) -> Vec3 {
        self.stats.pixels_resolved += 1;

        let mv = self.motion_vectors.get(x, y);

        // Compute reprojected UV.
        let u = (x as f32 + 0.5) / self.width as f32;
        let v = (y as f32 + 0.5) / self.height as f32;
        let prev_u = u - mv.uv_delta.x;
        let prev_v = v - mv.uv_delta.y;

        // Check if reprojected UV is out of bounds.
        let out_of_bounds = prev_u < 0.0 || prev_u > 1.0 || prev_v < 0.0 || prev_v > 1.0;

        // Sample history.
        let history = if out_of_bounds || self.cut_detector.should_reset() {
            self.stats.pixels_reset += 1;
            AccumulationPixel {
                color: current_color,
                depth: current_depth,
                frame_count: 0,
                valid: false,
            }
        } else {
            self.accumulation.sample_history_bilinear(prev_u, prev_v)
        };

        // Disocclusion check.
        if mv.disoccluded {
            self.stats.pixels_disoccluded += 1;
        }

        // Determine blend factor.
        let mut blend = self.config.blend_factor;

        // Velocity rejection.
        let vel_mag = mv.uv_delta.length() * self.width as f32;
        let vel_weight = self.config.velocity_rejection.compute_weight(vel_mag);
        if vel_weight < 1.0 {
            self.stats.pixels_velocity_rejected += 1;
        }
        blend *= vel_weight;

        // Motion vector confidence.
        blend *= mv.confidence;

        // Depth rejection.
        if self.config.depth_rejection_enabled && history.valid {
            let depth_diff = (current_depth - history.depth).abs();
            let relative_diff = depth_diff / current_depth.max(0.001);
            if relative_diff > self.config.depth_rejection_threshold {
                blend *= (1.0 - (relative_diff / self.config.depth_rejection_threshold).min(1.0)).max(0.0);
                self.stats.pixels_depth_rejected += 1;
            }
        }

        // Not valid history -> use current only.
        if !history.valid || self.cut_detector.should_reset() {
            blend = 0.0;
        }

        // Responsive factor: reduce blend for high luminance difference.
        if self.config.responsive_factor > 0.0 && history.valid {
            let lum_current = Self::luminance(current_color);
            let lum_history = Self::luminance(history.color);
            let lum_diff = (lum_current - lum_history).abs();
            let responsive_reduce = (lum_diff * self.config.responsive_factor).min(1.0);
            blend *= 1.0 - responsive_reduce;
        }

        // Clamp blend factor.
        blend = blend.clamp(MIN_BLEND_FACTOR, MAX_BLEND_FACTOR);

        // Apply tonemapping if configured.
        let (current_work, history_work) = if self.config.tonemap_before_blend {
            (Self::tonemap(current_color), Self::tonemap(history.color))
        } else {
            (current_color, history.color)
        };

        // Neighborhood clamping.
        let clamped_history = if !neighborhood_samples.is_empty()
            && self.config.clamping_mode != ClampingMode::None
        {
            let samples_work: Vec<Vec3> = if self.config.tonemap_before_blend {
                neighborhood_samples.iter().map(|s| Self::tonemap(*s)).collect()
            } else {
                neighborhood_samples.to_vec()
            };

            let neighborhood = Self::compute_neighborhood(&samples_work, self.config.variance_gamma);
            let clamped = neighborhood.clamp_color(history_work, self.config.clamping_mode);
            if clamped.sub(history_work).length_sq() > 1e-6 {
                self.stats.pixels_clamped += 1;
            }
            clamped
        } else {
            history_work
        };

        // Blend current and clamped history.
        let blended = current_work.scale(1.0 - blend).add(clamped_history.scale(blend));

        // Inverse tonemap if needed.
        let result = if self.config.tonemap_before_blend {
            Self::tonemap_inverse(blended)
        } else {
            blended
        };

        // Store in current accumulation buffer.
        let new_frame_count = if history.valid {
            history.frame_count.saturating_add(1).min(CONVERGENCE_FRAME_COUNT as u16)
        } else {
            1
        };

        self.accumulation.set_current(x, y, AccumulationPixel {
            color: result,
            depth: current_depth,
            frame_count: new_frame_count,
            valid: true,
        });

        result
    }

    /// Compute neighborhood result from samples.
    fn compute_neighborhood(samples: &[Vec3], gamma: f32) -> NeighborhoodResult {
        let mut aabb = ColorAabb::empty();
        for s in samples {
            aabb.expand(*s);
        }
        let variance_clip = VarianceClipRegion::compute(samples, gamma);

        NeighborhoodResult {
            aabb,
            variance_clip,
            samples: samples.to_vec(),
            sample_count: samples.len(),
        }
    }

    /// Get the resolved color at a pixel from the current accumulation buffer.
    pub fn get_resolved(&self, x: u32, y: u32) -> Vec3 {
        self.accumulation.get_current(x, y).color
    }

    /// Get the frame count at a pixel (convergence measure).
    pub fn get_convergence(&self, x: u32, y: u32) -> u16 {
        self.accumulation.get_current(x, y).frame_count
    }

    /// Get statistics from the last frame.
    pub fn stats(&self) -> &TemporalFilterStats {
        &self.stats
    }

    /// Finalize frame: compute aggregate statistics.
    pub fn end_frame(&mut self) {
        if self.stats.pixels_resolved > 0 {
            // Compute average blend factor (approximate from stats).
            let rejected = self.stats.pixels_velocity_rejected
                + self.stats.pixels_depth_rejected
                + self.stats.pixels_reset;
            let valid = self.stats.pixels_resolved.saturating_sub(rejected);
            self.stats.avg_blend_factor = if valid > 0 {
                self.config.blend_factor * (valid as f32 / self.stats.pixels_resolved as f32)
            } else {
                0.0
            };

            // Compute average convergence.
            let mut total_convergence: u64 = 0;
            let mut valid_count: u64 = 0;
            let pixel_count = self.accumulation.pixel_count();
            // Sample a subset to avoid iterating all pixels.
            let step = (pixel_count / 1000).max(1);
            for i in (0..pixel_count).step_by(step) {
                let x = (i as u32) % self.width;
                let y = (i as u32) / self.width;
                let p = self.accumulation.get_current(x, y);
                if p.valid {
                    total_convergence += p.frame_count as u64;
                    valid_count += 1;
                }
            }
            self.stats.avg_convergence = if valid_count > 0 {
                total_convergence as f32 / valid_count as f32
            } else {
                0.0
            };
        }
    }

    /// Force a complete reset (e.g. on scene load).
    pub fn force_reset(&mut self) {
        self.accumulation.clear();
        self.motion_vectors.clear();
        self.jitter.reset();
        self.cut_detector.reset();
        self.frame_count = 0;
        self.stats = TemporalFilterStats::default();
    }

    /// Get the total number of frames processed.
    pub fn total_frames(&self) -> u64 {
        self.frame_count
    }

    /// Get the current jitter sequence.
    pub fn jitter_sequence(&self) -> &JitterSequence {
        &self.jitter
    }

    /// Get the accumulation buffer (read only).
    pub fn accumulation_buffer(&self) -> &AccumulationBuffer {
        &self.accumulation
    }

    /// Get mutable access to the accumulation buffer.
    pub fn accumulation_buffer_mut(&mut self) -> &mut AccumulationBuffer {
        &mut self.accumulation
    }

    /// Get the width of the filter.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the height of the filter.
    pub fn height(&self) -> u32 {
        self.height
    }
}

// ---------------------------------------------------------------------------
// Reprojection utilities
// ---------------------------------------------------------------------------

/// Reproject a world-space position from current frame to previous frame UV.
pub fn reproject_world_to_prev_uv(
    world_pos: Vec3,
    prev_view_proj: &Mat4,
    viewport_width: u32,
    viewport_height: u32,
) -> Option<Vec2> {
    let clip = prev_view_proj.transform_point(world_pos);
    // clip.x and clip.y are in [-1, 1] after perspective divide.
    let u = (clip.x + 1.0) * 0.5;
    let v = (1.0 - clip.y) * 0.5; // Flip Y for screen space.

    if u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0 {
        Some(Vec2::new(u, v))
    } else {
        None
    }
}

/// Compute motion vector from current and previous view-projection matrices.
pub fn compute_motion_vector(
    world_pos: Vec3,
    current_view_proj: &Mat4,
    prev_view_proj: &Mat4,
) -> Vec2 {
    let curr_clip = current_view_proj.transform_point(world_pos);
    let prev_clip = prev_view_proj.transform_point(world_pos);

    let curr_uv = Vec2::new((curr_clip.x + 1.0) * 0.5, (1.0 - curr_clip.y) * 0.5);
    let prev_uv = Vec2::new((prev_clip.x + 1.0) * 0.5, (1.0 - prev_clip.y) * 0.5);

    curr_uv.sub(prev_uv)
}

// ---------------------------------------------------------------------------
// Sharpen filter
// ---------------------------------------------------------------------------

/// Simple CAS-like sharpening pass applied after temporal resolve.
pub struct SharpenPass {
    strength: f32,
}

impl SharpenPass {
    /// Create a sharpening pass with the given strength (0..1).
    pub fn new(strength: f32) -> Self {
        Self {
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Set sharpening strength.
    pub fn set_strength(&mut self, strength: f32) {
        self.strength = strength.clamp(0.0, 1.0);
    }

    /// Get sharpening strength.
    pub fn strength(&self) -> f32 {
        self.strength
    }

    /// Apply sharpening to a single pixel given its 4 neighbors.
    pub fn sharpen_pixel(
        &self,
        center: Vec3,
        left: Vec3,
        right: Vec3,
        up: Vec3,
        down: Vec3,
    ) -> Vec3 {
        if self.strength <= 0.0 {
            return center;
        }

        // Compute the neighborhood average.
        let avg = left.add(right).add(up).add(down).scale(0.25);

        // Sharpen: center + strength * (center - average).
        let diff = center.sub(avg);
        let sharpened = center.add(diff.scale(self.strength));

        // Clamp to avoid negative values.
        Vec3::new(
            sharpened.x.max(0.0),
            sharpened.y.max(0.0),
            sharpened.z.max(0.0),
        )
    }
}

// ---------------------------------------------------------------------------
// Temporal filter presets
// ---------------------------------------------------------------------------

/// Predefined quality presets for temporal filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalFilterPreset {
    /// Minimal filtering, low latency.
    Low,
    /// Balanced quality and performance.
    Medium,
    /// High quality with variance clamping.
    High,
    /// Maximum quality with all features enabled.
    Ultra,
    /// Optimized for VR (low latency, minimal ghosting).
    VR,
}

impl TemporalFilterPreset {
    /// Convert a preset to a full configuration.
    pub fn to_config(self) -> TemporalFilterConfig {
        match self {
            Self::Low => TemporalFilterConfig {
                blend_factor: 0.8,
                clamping_mode: ClampingMode::Aabb,
                variance_gamma: 1.5,
                jitter_pattern: JitterPattern::Halton2,
                velocity_rejection: VelocityRejectionConfig {
                    mode: VelocityRejectionMode::HardReject,
                    threshold: 20.0,
                    ..Default::default()
                },
                depth_rejection_enabled: false,
                tonemap_before_blend: false,
                sharpen_output: false,
                ..Default::default()
            },
            Self::Medium => TemporalFilterConfig {
                blend_factor: 0.9,
                clamping_mode: ClampingMode::VarianceClip,
                variance_gamma: 1.25,
                jitter_pattern: JitterPattern::Halton4,
                velocity_rejection: VelocityRejectionConfig::default(),
                depth_rejection_enabled: true,
                depth_rejection_threshold: 0.1,
                tonemap_before_blend: true,
                sharpen_output: false,
                ..Default::default()
            },
            Self::High => TemporalFilterConfig {
                blend_factor: 0.95,
                clamping_mode: ClampingMode::VarianceClip,
                variance_gamma: 1.0,
                jitter_pattern: JitterPattern::Halton8,
                velocity_rejection: VelocityRejectionConfig {
                    mode: VelocityRejectionMode::SoftFalloff,
                    threshold: 40.0,
                    falloff_exponent: 2.0,
                    ..Default::default()
                },
                depth_rejection_enabled: true,
                depth_rejection_threshold: 0.05,
                tonemap_before_blend: true,
                sharpen_output: true,
                sharpen_strength: 0.15,
                responsive_factor: 0.1,
                ..Default::default()
            },
            Self::Ultra => TemporalFilterConfig {
                blend_factor: 0.97,
                clamping_mode: ClampingMode::AabbVarianceIntersection,
                variance_gamma: 0.75,
                jitter_pattern: JitterPattern::Halton16,
                velocity_rejection: VelocityRejectionConfig {
                    mode: VelocityRejectionMode::AdaptiveConfidence,
                    threshold: 60.0,
                    confidence_min: 0.05,
                    ..Default::default()
                },
                depth_rejection_enabled: true,
                depth_rejection_threshold: 0.02,
                tonemap_before_blend: true,
                sharpen_output: true,
                sharpen_strength: 0.2,
                responsive_factor: 0.15,
                neighborhood_pattern: NeighborhoodPattern::Wide5x5,
                ..Default::default()
            },
            Self::VR => TemporalFilterConfig {
                blend_factor: 0.75,
                clamping_mode: ClampingMode::Aabb,
                variance_gamma: 2.0,
                jitter_pattern: JitterPattern::Halton2,
                velocity_rejection: VelocityRejectionConfig {
                    mode: VelocityRejectionMode::HardReject,
                    threshold: 10.0,
                    ..Default::default()
                },
                depth_rejection_enabled: true,
                depth_rejection_threshold: 0.1,
                tonemap_before_blend: false,
                sharpen_output: false,
                responsive_factor: 0.3,
                ..Default::default()
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Temporal filter history (for debugging)
// ---------------------------------------------------------------------------

/// Record of a single frame's filter state for debugging.
#[derive(Debug, Clone)]
pub struct TemporalFilterFrameRecord {
    /// Frame number.
    pub frame: u64,
    /// Stats snapshot.
    pub stats: TemporalFilterStats,
    /// Jitter offset used.
    pub jitter_offset: Vec2,
}

/// Rolling history of filter state for debugging purposes.
pub struct TemporalFilterHistory {
    records: VecDeque<TemporalFilterFrameRecord>,
    max_records: usize,
}

impl TemporalFilterHistory {
    /// Create a history buffer that stores the last N frame records.
    pub fn new(max_records: usize) -> Self {
        Self {
            records: VecDeque::with_capacity(max_records),
            max_records,
        }
    }

    /// Record a new frame.
    pub fn record(&mut self, frame: u64, stats: TemporalFilterStats, jitter: Vec2) {
        if self.records.len() >= self.max_records {
            self.records.pop_front();
        }
        self.records.push_back(TemporalFilterFrameRecord {
            frame,
            stats,
            jitter_offset: jitter,
        });
    }

    /// Get all recorded frames.
    pub fn records(&self) -> &VecDeque<TemporalFilterFrameRecord> {
        &self.records
    }

    /// Get the most recent record.
    pub fn latest(&self) -> Option<&TemporalFilterFrameRecord> {
        self.records.back()
    }

    /// Clear all records.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Get the average blend factor over all recorded frames.
    pub fn avg_blend_factor(&self) -> f32 {
        if self.records.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.records.iter().map(|r| r.stats.avg_blend_factor).sum();
        sum / self.records.len() as f32
    }

    /// Get the number of camera cuts in the recorded history.
    pub fn camera_cut_count(&self) -> usize {
        self.records
            .iter()
            .filter(|r| r.stats.camera_cut_detected)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halton_sequence() {
        let h2 = halton(1, 2);
        assert!((h2 - 0.5).abs() < 1e-6);

        let h3 = halton(1, 3);
        assert!((h3 - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_jitter_sequence_cycles() {
        let mut jitter = JitterSequence::new(JitterPattern::Halton4);
        assert_eq!(jitter.sample_count(), 4);

        let offsets: Vec<Vec2> = (0..4).map(|_| jitter.next_offset()).collect();
        // Should cycle back.
        let after_cycle = jitter.next_offset();
        assert_eq!(after_cycle.x, offsets[0].x);
        assert_eq!(after_cycle.y, offsets[0].y);
    }

    #[test]
    fn test_velocity_rejection_hard() {
        let config = VelocityRejectionConfig {
            mode: VelocityRejectionMode::HardReject,
            threshold: 10.0,
            ..Default::default()
        };

        assert_eq!(config.compute_weight(5.0), 1.0);
        assert_eq!(config.compute_weight(15.0), 0.0);
    }

    #[test]
    fn test_velocity_rejection_soft() {
        let config = VelocityRejectionConfig {
            mode: VelocityRejectionMode::SoftFalloff,
            threshold: 10.0,
            falloff_exponent: 1.0,
            ..Default::default()
        };

        assert!((config.compute_weight(0.0) - 1.0).abs() < 1e-6);
        assert!((config.compute_weight(5.0) - 0.5).abs() < 1e-6);
        assert!((config.compute_weight(10.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_camera_cut_detection() {
        let config = CameraCutConfig {
            distance_threshold: 10.0,
            rotation_threshold: 0.5,
            enabled: true,
            reset_hold_frames: 2,
        };
        let mut detector = CameraCutDetector::new(config);

        let cam1 = CameraState {
            position: Vec3::ZERO,
            forward: Vec3::new(0.0, 0.0, -1.0),
            ..Default::default()
        };
        // First frame is always a cut.
        assert!(detector.update(&cam1));

        // Small movement: no cut.
        let cam2 = CameraState {
            position: Vec3::new(1.0, 0.0, 0.0),
            forward: Vec3::new(0.0, 0.0, -1.0),
            ..Default::default()
        };
        assert!(!detector.update(&cam2));

        // Teleport: cut.
        let cam3 = CameraState {
            position: Vec3::new(100.0, 0.0, 0.0),
            forward: Vec3::new(0.0, 0.0, -1.0),
            ..Default::default()
        };
        assert!(detector.update(&cam3));
    }

    #[test]
    fn test_color_aabb_clamp() {
        let mut aabb = ColorAabb::empty();
        aabb.expand(Vec3::new(0.0, 0.0, 0.0));
        aabb.expand(Vec3::new(1.0, 1.0, 1.0));

        let clamped = aabb.clamp_color(Vec3::new(1.5, -0.5, 0.5));
        assert!((clamped.x - 1.0).abs() < 1e-6);
        assert!((clamped.y - 0.0).abs() < 1e-6);
        assert!((clamped.z - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_variance_clip() {
        let samples = vec![
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.6, 0.4, 0.6),
            Vec3::new(0.4, 0.6, 0.4),
            Vec3::new(0.55, 0.45, 0.55),
        ];

        let region = VarianceClipRegion::compute(&samples, 1.0);
        // Mean should be approximately (0.5125, 0.4875, 0.5125).
        assert!((region.mean.x - 0.5125).abs() < 1e-4);
    }

    #[test]
    fn test_motion_vector_buffer() {
        let mut buf = MotionVectorBuffer::new(4, 4);
        buf.set(2, 3, MotionVector {
            uv_delta: Vec2::new(0.1, 0.2),
            confidence: 0.9,
            disoccluded: false,
        });

        let mv = buf.get(2, 3);
        assert!((mv.uv_delta.x - 0.1).abs() < 1e-6);
        assert!((mv.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_accumulation_buffer_swap() {
        let mut buf = AccumulationBuffer::new(2, 2);
        buf.set_current(0, 0, AccumulationPixel {
            color: Vec3::new(1.0, 0.0, 0.0),
            depth: 0.5,
            frame_count: 1,
            valid: true,
        });

        buf.swap();

        // After swap, the old current is now history.
        let hist = buf.get_history(0, 0);
        assert!((hist.color.x - 1.0).abs() < 1e-6);
        assert!(hist.valid);
    }

    #[test]
    fn test_temporal_filter_preset_configs() {
        let config = TemporalFilterPreset::High.to_config();
        assert!(config.blend_factor > 0.9);
        assert_eq!(config.clamping_mode, ClampingMode::VarianceClip);
        assert!(config.sharpen_output);
    }

    #[test]
    fn test_sharpen_pass() {
        let pass = SharpenPass::new(0.5);
        let center = Vec3::new(1.0, 1.0, 1.0);
        let neighbor = Vec3::new(0.8, 0.8, 0.8);
        let result = pass.sharpen_pixel(center, neighbor, neighbor, neighbor, neighbor);
        // Sharpened should be brighter than center since center > average.
        assert!(result.x > center.x);
    }

    #[test]
    fn test_compute_motion_vector() {
        let vp = Mat4::IDENTITY;
        let mv = compute_motion_vector(Vec3::ZERO, &vp, &vp);
        assert!((mv.x).abs() < 1e-6);
        assert!((mv.y).abs() < 1e-6);
    }
}
