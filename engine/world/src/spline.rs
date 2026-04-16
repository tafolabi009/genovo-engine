//! Spline and Path Tools
//!
//! Provides cubic spline paths for entity movement, mesh extrusion (roads,
//! rivers, pipes), instance scattering, and spatial queries. Splines are
//! represented as sequences of control points with tangent handles and
//! evaluated using Catmull-Rom or Hermite interpolation.
//!
//! # Evaluation
//!
//! Splines are parameterized by `t` in the range `[0, 1]` where `t=0` is the
//! first control point and `t=1` is the last. Arc-length reparameterization
//! is available for uniform-speed traversal.
//!
//! # ECS Integration
//!
//! The [`SplineComponent`] attaches a spline to an entity. The
//! [`SplineFollower`] moves an entity along a spline at a configurable speed.

use std::collections::HashMap;

use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use genovo_ecs::{Component, Entity};

// ---------------------------------------------------------------------------
// SplinePoint
// ---------------------------------------------------------------------------

/// A control point on a spline, with position and tangent handles.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SplinePoint {
    /// Position in world space.
    pub position: Vec3,
    /// Incoming tangent (handle direction relative to position).
    pub tangent_in: Vec3,
    /// Outgoing tangent (handle direction relative to position).
    pub tangent_out: Vec3,
    /// Rotation at this point (used for mesh extrusion orientation).
    pub rotation: Quat,
    /// Scale at this point (used for mesh extrusion profile scaling).
    pub scale: Vec2,
    /// Roll angle in radians (twist around the forward axis).
    pub roll: f32,
}

impl SplinePoint {
    /// Create a spline point at a position with auto-computed tangents.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            tangent_in: Vec3::ZERO,
            tangent_out: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec2::ONE,
            roll: 0.0,
        }
    }

    /// Create a spline point with explicit tangents.
    pub fn with_tangents(position: Vec3, tangent_in: Vec3, tangent_out: Vec3) -> Self {
        Self {
            position,
            tangent_in,
            tangent_out,
            rotation: Quat::IDENTITY,
            scale: Vec2::ONE,
            roll: 0.0,
        }
    }

    /// Set symmetric tangent (same direction, opposite signs).
    pub fn with_tangent(mut self, tangent: Vec3) -> Self {
        self.tangent_in = -tangent;
        self.tangent_out = tangent;
        self
    }

    /// Set roll angle.
    pub fn with_roll(mut self, roll: f32) -> Self {
        self.roll = roll;
        self
    }

    /// Set scale.
    pub fn with_scale(mut self, scale: Vec2) -> Self {
        self.scale = scale;
        self
    }
}

// ---------------------------------------------------------------------------
// SplineType
// ---------------------------------------------------------------------------

/// Type of spline interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplineType {
    /// Catmull-Rom spline: passes through all control points.
    CatmullRom,
    /// Hermite spline: uses explicit tangents at each point.
    Hermite,
    /// Bezier spline: uses tangent handles as Bezier control points.
    Bezier,
    /// Linear interpolation between points.
    Linear,
}

// ---------------------------------------------------------------------------
// SplinePath
// ---------------------------------------------------------------------------

/// A series of control points defining a path through 3D space.
///
/// The spline supports several interpolation modes and provides arc-length
/// parameterization for uniform-speed traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplinePath {
    /// Control points.
    pub points: Vec<SplinePoint>,
    /// Interpolation type.
    pub spline_type: SplineType,
    /// Whether the spline forms a closed loop.
    pub closed: bool,
    /// Tension parameter for Catmull-Rom splines.
    pub tension: f32,
    /// Cached arc-length table for reparameterization.
    /// Each entry maps a uniform `t` value to the corresponding arc length.
    #[serde(skip)]
    arc_length_table: Vec<(f32, f32)>,
    /// Total arc length of the spline.
    #[serde(skip)]
    total_arc_length: f32,
    /// Number of samples used to build the arc-length table.
    #[serde(skip)]
    arc_length_samples: usize,
}

impl SplinePath {
    /// Create a new empty spline.
    pub fn new(spline_type: SplineType) -> Self {
        Self {
            points: Vec::new(),
            spline_type,
            closed: false,
            tension: 0.5,
            arc_length_table: Vec::new(),
            total_arc_length: 0.0,
            arc_length_samples: 256,
        }
    }

    /// Create a spline from a list of positions (auto-compute tangents).
    pub fn from_positions(positions: &[Vec3], spline_type: SplineType) -> Self {
        let mut spline = Self::new(spline_type);
        for pos in positions {
            spline.add_point(SplinePoint::new(*pos));
        }
        spline.auto_compute_tangents();
        spline.rebuild_arc_length_table();
        spline
    }

    /// Add a control point.
    pub fn add_point(&mut self, point: SplinePoint) {
        self.points.push(point);
        self.invalidate_cache();
    }

    /// Insert a control point at index.
    pub fn insert_point(&mut self, index: usize, point: SplinePoint) {
        self.points.insert(index, point);
        self.invalidate_cache();
    }

    /// Remove a control point by index.
    pub fn remove_point(&mut self, index: usize) -> SplinePoint {
        let p = self.points.remove(index);
        self.invalidate_cache();
        p
    }

    /// Get the number of control points.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Set closed loop mode.
    pub fn set_closed(&mut self, closed: bool) {
        self.closed = closed;
        self.invalidate_cache();
    }

    /// Invalidate cached data (call after modifying points).
    fn invalidate_cache(&mut self) {
        self.arc_length_table.clear();
        self.total_arc_length = 0.0;
    }

    /// Auto-compute tangents from control point positions.
    ///
    /// For Catmull-Rom splines, tangents are derived from neighboring points.
    pub fn auto_compute_tangents(&mut self) {
        let n = self.points.len();
        if n < 2 {
            return;
        }

        for i in 0..n {
            let prev = if i > 0 {
                self.points[i - 1].position
            } else if self.closed {
                self.points[n - 1].position
            } else {
                self.points[0].position
            };

            let next = if i < n - 1 {
                self.points[i + 1].position
            } else if self.closed {
                self.points[0].position
            } else {
                self.points[n - 1].position
            };

            let tangent = (next - prev) * self.tension;
            self.points[i].tangent_in = -tangent;
            self.points[i].tangent_out = tangent;
        }
    }

    // -- Evaluation --------------------------------------------------------

    /// Evaluate the spline at parameter `t` (0.0 to 1.0).
    ///
    /// Returns the world-space position on the spline.
    pub fn evaluate(&self, t: f32) -> Vec3 {
        let n = self.points.len();
        if n == 0 {
            return Vec3::ZERO;
        }
        if n == 1 {
            return self.points[0].position;
        }

        let t_clamped = t.clamp(0.0, 1.0);
        let segment_count = if self.closed { n } else { n - 1 };
        let total_t = t_clamped * segment_count as f32;
        let segment = (total_t.floor() as usize).min(segment_count - 1);
        let local_t = total_t - segment as f32;

        let i0 = segment;
        let i1 = if self.closed {
            (segment + 1) % n
        } else {
            (segment + 1).min(n - 1)
        };

        match self.spline_type {
            SplineType::Linear => {
                let p0 = self.points[i0].position;
                let p1 = self.points[i1].position;
                Vec3::lerp(p0, p1, local_t)
            }
            SplineType::CatmullRom | SplineType::Hermite => {
                self.evaluate_hermite(i0, i1, local_t)
            }
            SplineType::Bezier => {
                self.evaluate_bezier(i0, i1, local_t)
            }
        }
    }

    /// Hermite interpolation between two control points.
    fn evaluate_hermite(&self, i0: usize, i1: usize, t: f32) -> Vec3 {
        let p0 = self.points[i0].position;
        let p1 = self.points[i1].position;
        let m0 = self.points[i0].tangent_out;
        let m1 = self.points[i1].tangent_in;

        let t2 = t * t;
        let t3 = t2 * t;

        // Hermite basis functions.
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
    }

    /// Bezier interpolation between two control points.
    fn evaluate_bezier(&self, i0: usize, i1: usize, t: f32) -> Vec3 {
        let p0 = self.points[i0].position;
        let p3 = self.points[i1].position;
        let p1 = p0 + self.points[i0].tangent_out;
        let p2 = p3 + self.points[i1].tangent_in;

        let omt = 1.0 - t;
        let omt2 = omt * omt;
        let omt3 = omt2 * omt;
        let t2 = t * t;
        let t3 = t2 * t;

        p0 * omt3 + p1 * (3.0 * omt2 * t) + p2 * (3.0 * omt * t2) + p3 * t3
    }

    /// Evaluate the tangent (first derivative) at parameter `t`.
    pub fn evaluate_tangent(&self, t: f32) -> Vec3 {
        let delta = 0.001;
        let t0 = (t - delta).max(0.0);
        let t1 = (t + delta).min(1.0);

        let p0 = self.evaluate(t0);
        let p1 = self.evaluate(t1);

        (p1 - p0).normalize_or_zero()
    }

    /// Evaluate the up vector at parameter `t`, accounting for roll.
    pub fn evaluate_up(&self, t: f32) -> Vec3 {
        let tangent = self.evaluate_tangent(t);

        // Default up is world Y.
        let base_up = if tangent.y.abs() > 0.99 {
            Vec3::X // Avoid degenerate case when tangent is parallel to Y.
        } else {
            Vec3::Y
        };

        let right = tangent.cross(base_up).normalize();
        let up = right.cross(tangent).normalize();

        // Apply roll by rotating around the tangent axis.
        let n = self.points.len();
        if n == 0 {
            return up;
        }

        let t_clamped = t.clamp(0.0, 1.0);
        let segment_count = if self.closed { n } else { n.saturating_sub(1) };
        if segment_count == 0 {
            return up;
        }
        let total_t = t_clamped * segment_count as f32;
        let segment = (total_t.floor() as usize).min(segment_count - 1);
        let local_t = total_t - segment as f32;

        let i0 = segment;
        let i1 = if self.closed {
            (segment + 1) % n
        } else {
            (segment + 1).min(n - 1)
        };

        let roll = self.points[i0].roll * (1.0 - local_t) + self.points[i1].roll * local_t;
        if roll.abs() > 1e-6 {
            let rot = Quat::from_axis_angle(tangent, roll);
            rot * up
        } else {
            up
        }
    }

    /// Get a transform (position + rotation) at parameter `t`.
    pub fn evaluate_transform(&self, t: f32) -> (Vec3, Quat) {
        let pos = self.evaluate(t);
        let tangent = self.evaluate_tangent(t);
        let up = self.evaluate_up(t);

        if tangent.length_squared() < 1e-6 {
            return (pos, Quat::IDENTITY);
        }

        let forward = tangent;
        let right = forward.cross(up).normalize();
        let corrected_up = right.cross(forward).normalize();

        let mat = glam::Mat3::from_cols(right, corrected_up, -forward);
        let rotation = Quat::from_mat3(&mat);

        (pos, rotation)
    }

    // -- Arc-length parameterization ---------------------------------------

    /// Rebuild the arc-length lookup table.
    ///
    /// Call this after modifying control points if you need arc-length
    /// reparameterization.
    pub fn rebuild_arc_length_table(&mut self) {
        let samples = self.arc_length_samples;
        self.arc_length_table.clear();
        self.arc_length_table.reserve(samples + 1);

        let mut total = 0.0f32;
        let mut prev = self.evaluate(0.0);
        self.arc_length_table.push((0.0, 0.0));

        for i in 1..=samples {
            let t = i as f32 / samples as f32;
            let pos = self.evaluate(t);
            total += (pos - prev).length();
            self.arc_length_table.push((t, total));
            prev = pos;
        }

        self.total_arc_length = total;
    }

    /// Total arc length of the spline.
    pub fn arc_length(&self) -> f32 {
        self.total_arc_length
    }

    /// Convert a distance along the spline to a `t` parameter.
    ///
    /// Uses binary search on the arc-length table.
    pub fn distance_to_t(&self, distance: f32) -> f32 {
        if self.arc_length_table.is_empty() || self.total_arc_length < 1e-6 {
            return 0.0;
        }

        let d = distance.clamp(0.0, self.total_arc_length);

        // Binary search for the segment containing `d`.
        let mut lo = 0usize;
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

        let segment_length = d1 - d0;
        if segment_length < 1e-8 {
            return t0;
        }

        let frac = (d - d0) / segment_length;
        t0 + (t1 - t0) * frac
    }

    /// Convert a `t` parameter to a distance along the spline.
    pub fn t_to_distance(&self, t: f32) -> f32 {
        if self.arc_length_table.is_empty() {
            return 0.0;
        }

        let t_clamped = t.clamp(0.0, 1.0);

        // Linear search (could be binary but table is usually small).
        let mut prev = self.arc_length_table[0];
        for &(tt, dd) in &self.arc_length_table[1..] {
            if tt >= t_clamped {
                let frac = if tt - prev.0 > 1e-8 {
                    (t_clamped - prev.0) / (tt - prev.0)
                } else {
                    0.0
                };
                return prev.1 + (dd - prev.1) * frac;
            }
            prev = (tt, dd);
        }

        self.total_arc_length
    }

    /// Evaluate the spline at a uniform distance along its length.
    pub fn evaluate_at_distance(&self, distance: f32) -> Vec3 {
        let t = self.distance_to_t(distance);
        self.evaluate(t)
    }

    // -- Closest-point query -----------------------------------------------

    /// Find the closest point on the spline to a world-space position.
    ///
    /// Returns `(t, position, distance)`.
    pub fn closest_point(&self, target: Vec3) -> (f32, Vec3, f32) {
        if self.points.is_empty() {
            return (0.0, Vec3::ZERO, f32::MAX);
        }

        let samples = 128;
        let mut best_t = 0.0f32;
        let mut best_dist_sq = f32::MAX;
        let mut best_pos = Vec3::ZERO;

        // Coarse pass.
        for i in 0..=samples {
            let t = i as f32 / samples as f32;
            let pos = self.evaluate(t);
            let dist_sq = (pos - target).length_squared();
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_t = t;
                best_pos = pos;
            }
        }

        // Refine with Newton-Raphson-like iterations.
        let dt = 1.0 / samples as f32;
        let mut lo = (best_t - dt).max(0.0);
        let mut hi = (best_t + dt).min(1.0);

        for _ in 0..8 {
            let mid = (lo + hi) * 0.5;
            let q1 = (lo + mid) * 0.5;
            let q3 = (mid + hi) * 0.5;

            let d1 = (self.evaluate(q1) - target).length_squared();
            let d3 = (self.evaluate(q3) - target).length_squared();

            if d1 < d3 {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        let final_t = (lo + hi) * 0.5;
        let final_pos = self.evaluate(final_t);
        let final_dist = (final_pos - target).length();

        (final_t, final_pos, final_dist)
    }

    /// Get the bounding box of the spline (approximate).
    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        if self.points.is_empty() {
            return (Vec3::ZERO, Vec3::ZERO);
        }

        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);

        // Sample the spline for bounds (control points alone are not sufficient
        // for curves).
        let samples = self.points.len() * 16;
        for i in 0..=samples {
            let t = i as f32 / samples as f32;
            let pos = self.evaluate(t);
            min = min.min(pos);
            max = max.max(pos);
        }

        (min, max)
    }
}

// ---------------------------------------------------------------------------
// SplineComponent
// ---------------------------------------------------------------------------

/// ECS component that attaches a spline to an entity.
#[derive(Debug, Clone)]
pub struct SplineComponent {
    /// The spline path.
    pub spline: SplinePath,
    /// Whether the spline is visible in the editor.
    pub visible: bool,
    /// Color for debug rendering.
    pub debug_color: [f32; 4],
    /// Whether the spline points have been modified since last rebuild.
    pub dirty: bool,
}

impl Component for SplineComponent {}

impl SplineComponent {
    /// Create a new spline component.
    pub fn new(spline: SplinePath) -> Self {
        Self {
            spline,
            visible: true,
            debug_color: [1.0, 1.0, 0.0, 1.0],
            dirty: false,
        }
    }
}

// ---------------------------------------------------------------------------
// SplineFollower
// ---------------------------------------------------------------------------

/// Moves an entity along a spline at a configurable speed.
///
/// The follower tracks a distance along the spline and updates the entity's
/// position and rotation each frame.
#[derive(Debug, Clone)]
pub struct SplineFollower {
    /// Entity that owns the spline.
    pub spline_entity: Entity,
    /// Entity being moved along the spline.
    pub follower_entity: Entity,
    /// Current distance along the spline.
    pub current_distance: f32,
    /// Movement speed in world units per second.
    pub speed: f32,
    /// Whether the follower should loop back to the start.
    pub looping: bool,
    /// Whether to reverse direction at the end (ping-pong).
    pub ping_pong: bool,
    /// Current direction: 1.0 = forward, -1.0 = backward.
    pub direction: f32,
    /// Whether the follower is currently active.
    pub active: bool,
    /// Whether to orient the entity along the spline tangent.
    pub orient_to_spline: bool,
    /// Offset from the spline position.
    pub offset: Vec3,
    /// Callback parameter: how far along (0..1) the follower is.
    pub progress: f32,
    /// Whether the follower has reached the end at least once.
    pub completed: bool,
}

impl Component for SplineFollower {}

impl SplineFollower {
    /// Create a new spline follower.
    pub fn new(spline_entity: Entity, follower_entity: Entity, speed: f32) -> Self {
        Self {
            spline_entity,
            follower_entity,
            current_distance: 0.0,
            speed,
            looping: false,
            ping_pong: false,
            direction: 1.0,
            active: true,
            orient_to_spline: true,
            offset: Vec3::ZERO,
            progress: 0.0,
            completed: false,
        }
    }

    /// Enable looping.
    pub fn with_looping(mut self) -> Self {
        self.looping = true;
        self
    }

    /// Enable ping-pong mode.
    pub fn with_ping_pong(mut self) -> Self {
        self.ping_pong = true;
        self
    }

    /// Update the follower's position along the spline.
    ///
    /// Returns `(position, rotation)` for the follower entity.
    pub fn update(&mut self, spline: &SplinePath, dt: f32) -> Option<(Vec3, Quat)> {
        if !self.active {
            return None;
        }

        let total_length = spline.arc_length();
        if total_length < 1e-6 {
            return None;
        }

        // Advance distance.
        self.current_distance += self.speed * self.direction * dt;

        // Handle end-of-spline.
        if self.current_distance > total_length {
            if self.ping_pong {
                self.direction = -1.0;
                self.current_distance = total_length * 2.0 - self.current_distance;
            } else if self.looping {
                self.current_distance = self.current_distance % total_length;
            } else {
                self.current_distance = total_length;
                self.active = false;
                self.completed = true;
            }
        } else if self.current_distance < 0.0 {
            if self.ping_pong {
                self.direction = 1.0;
                self.current_distance = -self.current_distance;
            } else if self.looping {
                self.current_distance = total_length + self.current_distance;
            } else {
                self.current_distance = 0.0;
                self.active = false;
                self.completed = true;
            }
        }

        self.progress = self.current_distance / total_length;

        // Evaluate position and rotation on the spline.
        let t = spline.distance_to_t(self.current_distance);
        let (mut pos, rot) = spline.evaluate_transform(t);
        pos += self.offset;

        if self.orient_to_spline {
            Some((pos, rot))
        } else {
            Some((pos, Quat::IDENTITY))
        }
    }
}

// ---------------------------------------------------------------------------
// SplineMesh
// ---------------------------------------------------------------------------

/// Configuration for extruding a 2D profile along a spline to create
/// geometry (roads, rivers, pipes, rails).
#[derive(Debug, Clone)]
pub struct SplineMesh {
    /// Entity that owns the spline.
    pub spline_entity: Entity,
    /// 2D profile points (in local XY plane, extruded along Z).
    pub profile: Vec<Vec2>,
    /// Whether the profile is closed (e.g., a pipe).
    pub profile_closed: bool,
    /// Number of segments along the spline length.
    pub segments_along: u32,
    /// UV tiling along the spline.
    pub uv_tile_u: f32,
    /// UV tiling across the profile.
    pub uv_tile_v: f32,
    /// Material handle for the generated mesh.
    pub material_handle: u64,
    /// Whether to generate collision geometry.
    pub generate_collision: bool,
    /// Whether the mesh needs to be rebuilt.
    pub dirty: bool,
}

impl Component for SplineMesh {}

impl SplineMesh {
    /// Create a new spline mesh.
    pub fn new(spline_entity: Entity, profile: Vec<Vec2>) -> Self {
        Self {
            spline_entity,
            profile,
            profile_closed: false,
            segments_along: 64,
            uv_tile_u: 1.0,
            uv_tile_v: 1.0,
            material_handle: 0,
            generate_collision: true,
            dirty: true,
        }
    }

    /// Create a flat road profile.
    pub fn road_profile(width: f32) -> Vec<Vec2> {
        let hw = width * 0.5;
        vec![
            Vec2::new(-hw, 0.0),
            Vec2::new(-hw * 0.8, 0.02),
            Vec2::new(hw * 0.8, 0.02),
            Vec2::new(hw, 0.0),
        ]
    }

    /// Create a circular pipe profile.
    pub fn pipe_profile(radius: f32, segments: u32) -> Vec<Vec2> {
        let mut profile = Vec::with_capacity(segments as usize);
        for i in 0..segments {
            let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
            profile.push(Vec2::new(angle.cos() * radius, angle.sin() * radius));
        }
        profile
    }

    /// Create a river/channel profile (concave).
    pub fn river_profile(width: f32, depth: f32) -> Vec<Vec2> {
        let hw = width * 0.5;
        vec![
            Vec2::new(-hw, 0.0),
            Vec2::new(-hw * 0.7, -depth * 0.5),
            Vec2::new(-hw * 0.3, -depth),
            Vec2::new(hw * 0.3, -depth),
            Vec2::new(hw * 0.7, -depth * 0.5),
            Vec2::new(hw, 0.0),
        ]
    }

    /// Generate vertex data for the extruded mesh.
    ///
    /// Returns `(positions, normals, uvs, indices)`.
    pub fn generate_mesh(&self, spline: &SplinePath) -> SplineMeshData {
        let n_along = self.segments_along as usize;
        let n_profile = self.profile.len();

        if n_profile < 2 || n_along < 1 {
            return SplineMeshData::empty();
        }

        let total_length = spline.arc_length();
        let vertex_count = (n_along + 1) * n_profile;
        let index_count = n_along * (n_profile - 1) * 6;

        let mut positions = Vec::with_capacity(vertex_count);
        let mut normals = Vec::with_capacity(vertex_count);
        let mut uvs = Vec::with_capacity(vertex_count);
        let mut indices = Vec::with_capacity(index_count);

        for seg in 0..=n_along {
            let t = seg as f32 / n_along as f32;
            let distance = t * total_length;
            let spline_t = spline.distance_to_t(distance);

            let pos = spline.evaluate(spline_t);
            let tangent = spline.evaluate_tangent(spline_t);
            let up = spline.evaluate_up(spline_t);

            let right = tangent.cross(up).normalize();
            let corrected_up = right.cross(tangent).normalize();

            // Get interpolated scale from spline points.
            let scale = self.interpolate_scale(spline, spline_t);

            for (pi, profile_pt) in self.profile.iter().enumerate() {
                let local_x = profile_pt.x * scale.x;
                let local_y = profile_pt.y * scale.y;

                let world_pos = pos + right * local_x + corrected_up * local_y;
                positions.push(world_pos);

                // Approximate normal by cross product of tangent derivatives.
                let normal = corrected_up; // Simplified: use up vector as normal.
                normals.push(normal);

                let u = (pi as f32 / (n_profile - 1) as f32) * self.uv_tile_u;
                let v = (distance / total_length.max(1.0)) * self.uv_tile_v;
                uvs.push(Vec2::new(u, v));
            }
        }

        // Generate triangle indices.
        for seg in 0..n_along {
            for pi in 0..(n_profile - 1) {
                let base = seg * n_profile + pi;
                let next_ring = (seg + 1) * n_profile + pi;

                // First triangle.
                indices.push(base as u32);
                indices.push(next_ring as u32);
                indices.push((base + 1) as u32);

                // Second triangle.
                indices.push((base + 1) as u32);
                indices.push(next_ring as u32);
                indices.push((next_ring + 1) as u32);
            }

            // Close the profile if needed.
            if self.profile_closed {
                let base = seg * n_profile + (n_profile - 1);
                let next_ring = (seg + 1) * n_profile + (n_profile - 1);
                let base_first = seg * n_profile;
                let next_ring_first = (seg + 1) * n_profile;

                indices.push(base as u32);
                indices.push(next_ring as u32);
                indices.push(base_first as u32);

                indices.push(base_first as u32);
                indices.push(next_ring as u32);
                indices.push(next_ring_first as u32);
            }
        }

        SplineMeshData {
            positions,
            normals,
            uvs,
            indices,
        }
    }

    /// Interpolate the scale at a spline parameter `t`.
    fn interpolate_scale(&self, spline: &SplinePath, t: f32) -> Vec2 {
        let n = spline.points.len();
        if n == 0 {
            return Vec2::ONE;
        }
        if n == 1 {
            return spline.points[0].scale;
        }

        let total_t = t * (n - 1) as f32;
        let i0 = (total_t.floor() as usize).min(n - 2);
        let frac = total_t - i0 as f32;

        let s0 = spline.points[i0].scale;
        let s1 = spline.points[i0 + 1].scale;

        Vec2::lerp(s0, s1, frac)
    }
}

// ---------------------------------------------------------------------------
// SplineMeshData
// ---------------------------------------------------------------------------

/// Generated mesh data from spline extrusion.
#[derive(Debug, Clone)]
pub struct SplineMeshData {
    /// Vertex positions.
    pub positions: Vec<Vec3>,
    /// Vertex normals.
    pub normals: Vec<Vec3>,
    /// Texture coordinates.
    pub uvs: Vec<Vec2>,
    /// Triangle indices.
    pub indices: Vec<u32>,
}

impl SplineMeshData {
    /// Create empty mesh data.
    pub fn empty() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

// ---------------------------------------------------------------------------
// SplineScatter
// ---------------------------------------------------------------------------

/// Scatters instances along or around a spline.
///
/// Used for placing vegetation, rocks, fence posts, or other repeated objects
/// along a path.
#[derive(Debug, Clone)]
pub struct SplineScatter {
    /// Entity that owns the spline.
    pub spline_entity: Entity,
    /// Mesh handle for the scattered instances.
    pub mesh_handle: u64,
    /// Material handle.
    pub material_handle: u64,
    /// Spacing between instances along the spline (world units).
    pub spacing: f32,
    /// Random offset range perpendicular to the spline.
    pub lateral_spread: f32,
    /// Random vertical offset range.
    pub vertical_spread: f32,
    /// Random scale range: (min_scale, max_scale).
    pub scale_range: (f32, f32),
    /// Random rotation range in radians around the Y axis.
    pub rotation_range: f32,
    /// Whether to align instances to the spline tangent.
    pub align_to_spline: bool,
    /// Whether to align instances to the terrain normal (if available).
    pub align_to_terrain: bool,
    /// Density multiplier (1.0 = one instance per `spacing` units).
    pub density: f32,
    /// Random seed for reproducible placement.
    pub seed: u64,
    /// Maximum number of instances.
    pub max_instances: usize,
    /// Generated instance transforms.
    pub instances: Vec<ScatterInstance>,
}

impl Component for SplineScatter {}

/// A single scattered instance.
#[derive(Debug, Clone)]
pub struct ScatterInstance {
    /// World-space position.
    pub position: Vec3,
    /// Rotation.
    pub rotation: Quat,
    /// Uniform scale.
    pub scale: f32,
    /// Distance along the spline where this instance was placed.
    pub spline_distance: f32,
}

impl SplineScatter {
    /// Create a new spline scatter.
    pub fn new(spline_entity: Entity, mesh_handle: u64, spacing: f32) -> Self {
        Self {
            spline_entity,
            mesh_handle,
            material_handle: 0,
            spacing,
            lateral_spread: 0.0,
            vertical_spread: 0.0,
            scale_range: (1.0, 1.0),
            rotation_range: 0.0,
            align_to_spline: true,
            align_to_terrain: false,
            density: 1.0,
            seed: 42,
            max_instances: 10000,
            instances: Vec::new(),
        }
    }

    /// Generate scatter instances along the spline.
    pub fn generate(&mut self, spline: &SplinePath) {
        self.instances.clear();

        let total_length = spline.arc_length();
        if total_length < 1e-6 || self.spacing < 1e-6 {
            return;
        }

        let effective_spacing = self.spacing / self.density.max(0.01);
        let count = (total_length / effective_spacing).ceil() as usize;
        let count = count.min(self.max_instances);

        // Simple pseudo-random number generator seeded by instance index + seed.
        let hash = |i: usize, channel: u32| -> f32 {
            let mut h = (i as u64).wrapping_mul(2654435761) ^ self.seed;
            h = h.wrapping_add((channel as u64).wrapping_mul(6364136223846793005));
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            // Map to [0, 1).
            (h & 0xFFFFFF) as f32 / 0xFFFFFF as f32
        };

        for i in 0..count {
            let dist = i as f32 * effective_spacing;
            let t = spline.distance_to_t(dist);

            let base_pos = spline.evaluate(t);
            let tangent = spline.evaluate_tangent(t);
            let up = spline.evaluate_up(t);

            let right = tangent.cross(up).normalize();

            // Random offsets.
            let lateral = (hash(i, 0) - 0.5) * 2.0 * self.lateral_spread;
            let vertical = (hash(i, 1) - 0.5) * 2.0 * self.vertical_spread;
            let offset = right * lateral + Vec3::Y * vertical;

            let position = base_pos + offset;

            // Random rotation.
            let yaw = (hash(i, 2) - 0.5) * 2.0 * self.rotation_range;
            let base_rot = if self.align_to_spline {
                let (_, rot) = spline.evaluate_transform(t);
                rot
            } else {
                Quat::IDENTITY
            };
            let rotation = base_rot * Quat::from_rotation_y(yaw);

            // Random scale.
            let scale = self.scale_range.0
                + hash(i, 3) * (self.scale_range.1 - self.scale_range.0);

            self.instances.push(ScatterInstance {
                position,
                rotation,
                scale,
                spline_distance: dist,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_spline() -> SplinePath {
        SplinePath::from_positions(
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(100.0, 0.0, 0.0),
                Vec3::new(200.0, 0.0, 100.0),
                Vec3::new(300.0, 0.0, 100.0),
            ],
            SplineType::CatmullRom,
        )
    }

    #[test]
    fn evaluate_endpoints() {
        let spline = simple_spline();
        let start = spline.evaluate(0.0);
        let end = spline.evaluate(1.0);

        assert!((start - Vec3::new(0.0, 0.0, 0.0)).length() < 1.0);
        assert!((end - Vec3::new(300.0, 0.0, 100.0)).length() < 1.0);
    }

    #[test]
    fn evaluate_midpoint() {
        let spline = SplinePath::from_positions(
            &[Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)],
            SplineType::Linear,
        );
        let mid = spline.evaluate(0.5);
        assert!((mid.x - 50.0).abs() < 1.0);
    }

    #[test]
    fn arc_length_positive() {
        let spline = simple_spline();
        assert!(spline.arc_length() > 0.0);
    }

    #[test]
    fn distance_to_t_round_trip() {
        let spline = simple_spline();
        let half_length = spline.arc_length() * 0.5;
        let t = spline.distance_to_t(half_length);
        let d = spline.t_to_distance(t);
        assert!((d - half_length).abs() < 2.0);
    }

    #[test]
    fn closest_point_on_straight_line() {
        let spline = SplinePath::from_positions(
            &[Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)],
            SplineType::Linear,
        );
        let (t, pos, dist) = spline.closest_point(Vec3::new(50.0, 10.0, 0.0));
        assert!((pos.x - 50.0).abs() < 2.0);
        assert!((dist - 10.0).abs() < 2.0);
        assert!(t > 0.4 && t < 0.6);
    }

    #[test]
    fn spline_mesh_generation() {
        let spline = SplinePath::from_positions(
            &[Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)],
            SplineType::Linear,
        );
        let mesh = SplineMesh::new(
            Entity::PLACEHOLDER,
            SplineMesh::road_profile(10.0),
        );
        let data = mesh.generate_mesh(&spline);
        assert!(data.vertex_count() > 0);
        assert!(data.triangle_count() > 0);
    }

    #[test]
    fn scatter_generation() {
        let spline = SplinePath::from_positions(
            &[Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)],
            SplineType::Linear,
        );
        let mut scatter = SplineScatter::new(Entity::PLACEHOLDER, 0, 10.0);
        scatter.generate(&spline);
        assert!(scatter.instances.len() > 5);
    }

    #[test]
    fn spline_tangent_direction() {
        let spline = SplinePath::from_positions(
            &[Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)],
            SplineType::Linear,
        );
        let tangent = spline.evaluate_tangent(0.5);
        // Tangent should point roughly in the +X direction.
        assert!(tangent.x > 0.9);
    }
}
