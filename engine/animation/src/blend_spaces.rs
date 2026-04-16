//! 1D and 2D blend spaces for parameter-driven animation blending.
//!
//! Blend spaces map gameplay parameters (movement speed, direction angle) to
//! weighted combinations of animation clips. Unlike the simple 1D blending in
//! [`BlendTree`](crate::blend_tree), these spaces support scatter-point layouts
//! with Delaunay triangulation for smooth 2D interpolation.
//!
//! # Overview
//!
//! - [`BlendSpace1D`]: blends along a single axis (e.g., speed 0..10).
//! - [`BlendSpace2D`]: blends over two axes using Delaunay triangulation
//!   and barycentric interpolation inside the containing triangle.
//! - [`DirectionalBlendSpace`]: convenience wrapper for the common
//!   direction + speed locomotion use-case.

use genovo_core::Transform;
use glam::{Quat, Vec2, Vec3};

use crate::skeleton::AnimationClip;

// ---------------------------------------------------------------------------
// BlendSpace1D
// ---------------------------------------------------------------------------

/// A 1D blend space that selects and blends between animations based on a
/// single scalar parameter.
///
/// Points are laid out along a single axis. At runtime the parameter value is
/// used to locate the two bracketing points and a linear blend weight is
/// computed between them.
#[derive(Debug, Clone)]
pub struct BlendSpace1D {
    /// Name of the blend space (for debugging / editor display).
    pub name: String,

    /// Sorted list of `(parameter_value, clip_index)` points.
    points: Vec<BlendPoint1D>,

    /// Current parameter value.
    parameter: f32,

    /// Current playback time (shared across all clips for synchronized blending).
    playback_time: f32,

    /// Whether to synchronize clip playback by normalized time.
    pub sync_playback: bool,
}

/// A single point in a 1D blend space.
#[derive(Debug, Clone)]
pub struct BlendPoint1D {
    /// Parameter value at which this animation is fully weighted.
    pub value: f32,
    /// Index into the external clip collection.
    pub clip_index: usize,
    /// Playback speed multiplier for this clip.
    pub speed: f32,
}

impl BlendSpace1D {
    /// Create a new 1D blend space.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            points: Vec::new(),
            parameter: 0.0,
            playback_time: 0.0,
            sync_playback: true,
        }
    }

    /// Add a point mapping a parameter value to a clip index.
    pub fn add_point(&mut self, value: f32, clip_index: usize) {
        self.points.push(BlendPoint1D {
            value,
            clip_index,
            speed: 1.0,
        });
        // Keep sorted by parameter value.
        self.points.sort_by(|a, b| {
            a.value
                .partial_cmp(&b.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Add a point with a custom playback speed.
    pub fn add_point_with_speed(&mut self, value: f32, clip_index: usize, speed: f32) {
        self.points.push(BlendPoint1D {
            value,
            clip_index,
            speed,
        });
        self.points.sort_by(|a, b| {
            a.value
                .partial_cmp(&b.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Set the current parameter value.
    pub fn set_parameter(&mut self, value: f32) {
        self.parameter = value;
    }

    /// Get the current parameter value.
    pub fn parameter(&self) -> f32 {
        self.parameter
    }

    /// Number of blend points.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Compute blend weights for the current parameter value.
    ///
    /// Returns a list of `(clip_index, weight)` pairs. At most two clips will
    /// have non-zero weight.
    pub fn compute_weights(&self) -> Vec<(usize, f32)> {
        if self.points.is_empty() {
            return Vec::new();
        }
        if self.points.len() == 1 {
            return vec![(self.points[0].clip_index, 1.0)];
        }

        let val = self.parameter;

        // Clamp to range.
        if val <= self.points[0].value {
            return vec![(self.points[0].clip_index, 1.0)];
        }
        let last = self.points.len() - 1;
        if val >= self.points[last].value {
            return vec![(self.points[last].clip_index, 1.0)];
        }

        // Binary search for the bracket.
        let idx = self.find_bracket(val);
        let lo = &self.points[idx];
        let hi = &self.points[idx + 1];

        let range = hi.value - lo.value;
        let t = if range.abs() > f32::EPSILON {
            (val - lo.value) / range
        } else {
            0.0
        };

        vec![
            (lo.clip_index, 1.0 - t),
            (hi.clip_index, t),
        ]
    }

    /// Evaluate the blend space and produce a blended pose.
    pub fn evaluate(
        &mut self,
        clips: &[AnimationClip],
        dt: f32,
        bone_count: usize,
    ) -> Vec<Transform> {
        // Advance playback time.
        let max_duration = clips
            .iter()
            .map(|c| c.duration)
            .fold(0.0f32, f32::max)
            .max(0.001);

        self.playback_time += dt;
        if self.playback_time > max_duration {
            self.playback_time = self.playback_time.rem_euclid(max_duration);
        }

        let weights = self.compute_weights();
        if weights.is_empty() {
            return vec![Transform::IDENTITY; bone_count];
        }

        // Fast path: single clip.
        if weights.len() == 1 {
            let (clip_idx, _) = weights[0];
            if clip_idx < clips.len() {
                let time = if self.sync_playback {
                    let norm = self.playback_time / max_duration;
                    norm * clips[clip_idx].duration
                } else {
                    self.playback_time
                };
                return clips[clip_idx].sample_pose(time, bone_count);
            }
            return vec![Transform::IDENTITY; bone_count];
        }

        // Two-clip blend.
        let (idx_a, weight_a) = weights[0];
        let (idx_b, weight_b) = weights[1];

        let pose_a = if idx_a < clips.len() {
            let time = if self.sync_playback {
                let norm = self.playback_time / max_duration;
                norm * clips[idx_a].duration
            } else {
                self.playback_time
            };
            clips[idx_a].sample_pose(time, bone_count)
        } else {
            vec![Transform::IDENTITY; bone_count]
        };

        let pose_b = if idx_b < clips.len() {
            let time = if self.sync_playback {
                let norm = self.playback_time / max_duration;
                norm * clips[idx_b].duration
            } else {
                self.playback_time
            };
            clips[idx_b].sample_pose(time, bone_count)
        } else {
            vec![Transform::IDENTITY; bone_count]
        };

        blend_poses(&pose_a, &pose_b, weight_b)
    }

    /// Binary search for the index of the lower-bracket point.
    fn find_bracket(&self, val: f32) -> usize {
        let mut lo = 0usize;
        let mut hi = self.points.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.points[mid].value <= val {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 { 0 } else { lo - 1 }
    }
}

// ---------------------------------------------------------------------------
// BlendSpace2D
// ---------------------------------------------------------------------------

/// A 2D blend space using Delaunay triangulation for smooth multi-animation
/// blending over two parameters.
///
/// Points are scattered in a 2D parameter space (e.g., direction on X,
/// speed on Y). A Delaunay triangulation is computed once on construction or
/// when points change. At evaluation time the containing triangle is found
/// and barycentric weights produce a smooth blend of exactly 3 animations.
#[derive(Debug, Clone)]
pub struct BlendSpace2D {
    /// Name of the blend space.
    pub name: String,

    /// Scatter points: `(parameter_position, clip_index)`.
    points: Vec<BlendPoint2D>,

    /// Delaunay triangulation: each triangle is a triple of point indices.
    triangles: Vec<[usize; 3]>,

    /// Current 2D parameter value.
    parameter: Vec2,

    /// Current playback time.
    playback_time: f32,

    /// Whether to sync clip playback.
    pub sync_playback: bool,

    /// Whether the triangulation needs to be rebuilt.
    dirty: bool,
}

/// A single point in a 2D blend space.
#[derive(Debug, Clone)]
pub struct BlendPoint2D {
    /// Position in the 2D parameter space.
    pub position: Vec2,
    /// Index into the external clip collection.
    pub clip_index: usize,
    /// Optional playback speed multiplier.
    pub speed: f32,
}

impl BlendSpace2D {
    /// Create a new empty 2D blend space.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            points: Vec::new(),
            triangles: Vec::new(),
            parameter: Vec2::ZERO,
            playback_time: 0.0,
            sync_playback: true,
            dirty: true,
        }
    }

    /// Add a point.
    pub fn add_point(&mut self, position: Vec2, clip_index: usize) {
        self.points.push(BlendPoint2D {
            position,
            clip_index,
            speed: 1.0,
        });
        self.dirty = true;
    }

    /// Add a point with a custom speed.
    pub fn add_point_with_speed(&mut self, position: Vec2, clip_index: usize, speed: f32) {
        self.points.push(BlendPoint2D {
            position,
            clip_index,
            speed,
        });
        self.dirty = true;
    }

    /// Set the current 2D parameter.
    pub fn set_parameter(&mut self, value: Vec2) {
        self.parameter = value;
    }

    /// Get the current parameter.
    pub fn parameter(&self) -> Vec2 {
        self.parameter
    }

    /// Number of scatter points.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Number of triangles in the current triangulation.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Force-rebuild the Delaunay triangulation.
    pub fn rebuild_triangulation(&mut self) {
        self.triangles = delaunay_triangulate(&self.points);
        self.dirty = false;
    }

    /// Ensure the triangulation is up-to-date.
    fn ensure_triangulation(&mut self) {
        if self.dirty {
            self.rebuild_triangulation();
        }
    }

    /// Compute blend weights for the current parameter value.
    ///
    /// Returns up to 3 `(clip_index, weight)` pairs from the containing
    /// triangle, or falls back to nearest-point if the parameter is outside
    /// the convex hull.
    pub fn compute_weights(&mut self) -> Vec<(usize, f32)> {
        self.ensure_triangulation();

        if self.points.is_empty() {
            return Vec::new();
        }
        if self.points.len() == 1 {
            return vec![(self.points[0].clip_index, 1.0)];
        }
        if self.points.len() == 2 {
            return self.blend_two_points();
        }
        if self.triangles.is_empty() {
            // Fallback: all points are collinear.
            return self.nearest_point_fallback();
        }

        let p = self.parameter;

        // Find the containing triangle.
        for tri in &self.triangles {
            let a = self.points[tri[0]].position;
            let b = self.points[tri[1]].position;
            let c = self.points[tri[2]].position;

            if let Some((u, v, w)) = barycentric(p, a, b, c) {
                if u >= -1e-4 && v >= -1e-4 && w >= -1e-4 {
                    // Inside (or on edge of) this triangle.
                    let mut weights = Vec::new();
                    if u > 1e-6 {
                        weights.push((self.points[tri[0]].clip_index, u));
                    }
                    if v > 1e-6 {
                        weights.push((self.points[tri[1]].clip_index, v));
                    }
                    if w > 1e-6 {
                        weights.push((self.points[tri[2]].clip_index, w));
                    }
                    // Normalize weights.
                    let total: f32 = weights.iter().map(|(_, w)| *w).sum();
                    if total > f32::EPSILON {
                        for entry in &mut weights {
                            entry.1 /= total;
                        }
                    }
                    return weights;
                }
            }
        }

        // Outside the convex hull: project onto nearest edge or use nearest point.
        self.nearest_edge_or_point()
    }

    /// Evaluate the blend space and produce a blended pose.
    pub fn evaluate(
        &mut self,
        clips: &[AnimationClip],
        dt: f32,
        bone_count: usize,
    ) -> Vec<Transform> {
        let max_duration = clips
            .iter()
            .map(|c| c.duration)
            .fold(0.0f32, f32::max)
            .max(0.001);

        self.playback_time += dt;
        if self.playback_time > max_duration {
            self.playback_time = self.playback_time.rem_euclid(max_duration);
        }

        let weights = self.compute_weights();
        if weights.is_empty() {
            return vec![Transform::IDENTITY; bone_count];
        }

        // Sample all weighted clips.
        let mut result = vec![Transform::IDENTITY; bone_count];
        let mut first = true;

        for &(clip_idx, weight) in &weights {
            if clip_idx >= clips.len() || weight < 1e-6 {
                continue;
            }
            let time = if self.sync_playback {
                let norm = self.playback_time / max_duration;
                norm * clips[clip_idx].duration
            } else {
                self.playback_time
            };
            let pose = clips[clip_idx].sample_pose(time, bone_count);

            if first {
                // Initialize with the first weighted pose.
                for i in 0..bone_count {
                    result[i] = Transform::new(
                        pose[i].position * weight,
                        pose[i].rotation, // Will be blended via successive slerps.
                        pose[i].scale * weight,
                    );
                }
                first = false;
            } else {
                // Accumulate: for position/scale we do weighted sum; for
                // rotation we do successive slerp.
                let blend_t = weight / (1.0 - weight + weight); // relative weight
                for i in 0..bone_count {
                    result[i].position += pose[i].position * weight;
                    result[i].scale += pose[i].scale * weight;
                    // Slerp towards this pose's rotation with relative weight.
                    result[i].rotation = result[i].rotation.slerp(pose[i].rotation, blend_t);
                }
            }
        }

        result
    }

    // ----- helpers -----

    /// Blend between two points using linear interpolation along their axis.
    fn blend_two_points(&self) -> Vec<(usize, f32)> {
        let a = self.points[0].position;
        let b = self.points[1].position;
        let ab = b - a;
        let len_sq = ab.length_squared();
        let t = if len_sq > f32::EPSILON {
            ((self.parameter - a).dot(ab) / len_sq).clamp(0.0, 1.0)
        } else {
            0.5
        };
        vec![
            (self.points[0].clip_index, 1.0 - t),
            (self.points[1].clip_index, t),
        ]
    }

    /// Nearest-point fallback when no triangulation is available.
    fn nearest_point_fallback(&self) -> Vec<(usize, f32)> {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for (i, pt) in self.points.iter().enumerate() {
            let d = (pt.position - self.parameter).length_squared();
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        vec![(self.points[best_idx].clip_index, 1.0)]
    }

    /// Project the parameter onto the nearest triangle edge when outside
    /// the convex hull.
    fn nearest_edge_or_point(&self) -> Vec<(usize, f32)> {
        let p = self.parameter;
        let mut best_dist = f32::MAX;
        let mut best_weights: Vec<(usize, f32)> = Vec::new();

        // Check each triangle edge.
        for tri in &self.triangles {
            for edge_idx in 0..3 {
                let i = tri[edge_idx];
                let j = tri[(edge_idx + 1) % 3];
                let a = self.points[i].position;
                let b = self.points[j].position;
                let (proj_t, dist) = project_onto_segment(p, a, b);

                if dist < best_dist {
                    best_dist = dist;
                    let t = proj_t.clamp(0.0, 1.0);
                    best_weights = vec![
                        (self.points[i].clip_index, 1.0 - t),
                        (self.points[j].clip_index, t),
                    ];
                }
            }
        }

        // Also check individual points.
        for (i, pt) in self.points.iter().enumerate() {
            let d = (pt.position - p).length();
            if d < best_dist {
                best_dist = d;
                best_weights = vec![(pt.clip_index, 1.0)];
            }
        }

        // Remove near-zero weights.
        best_weights.retain(|&(_, w)| w > 1e-6);
        if best_weights.is_empty() && !self.points.is_empty() {
            best_weights = vec![(self.points[0].clip_index, 1.0)];
        }

        best_weights
    }
}

// ---------------------------------------------------------------------------
// DirectionalBlendSpace
// ---------------------------------------------------------------------------

/// A specialized 2D blend space for direction + speed locomotion blending.
///
/// Pre-configured with cardinal and diagonal directions and an idle center
/// point. Clips are assigned to named slots (Forward, Backward, Left, Right,
/// etc.) and the blend space is automatically populated.
#[derive(Debug, Clone)]
pub struct DirectionalBlendSpace {
    /// The underlying 2D blend space.
    inner: BlendSpace2D,

    /// Clip index for idle (center point, speed = 0).
    pub idle_clip: Option<usize>,
    /// Clip index for forward movement.
    pub forward_clip: Option<usize>,
    /// Clip index for backward movement.
    pub backward_clip: Option<usize>,
    /// Clip index for strafe left.
    pub left_clip: Option<usize>,
    /// Clip index for strafe right.
    pub right_clip: Option<usize>,
    /// Clip index for forward-left diagonal.
    pub forward_left_clip: Option<usize>,
    /// Clip index for forward-right diagonal.
    pub forward_right_clip: Option<usize>,
    /// Clip index for backward-left diagonal.
    pub backward_left_clip: Option<usize>,
    /// Clip index for backward-right diagonal.
    pub backward_right_clip: Option<usize>,

    /// Maximum speed value (the radius of the blend space circle).
    pub max_speed: f32,
}

impl DirectionalBlendSpace {
    /// Create a new directional blend space with the given max speed.
    pub fn new(name: impl Into<String>, max_speed: f32) -> Self {
        Self {
            inner: BlendSpace2D::new(name),
            idle_clip: None,
            forward_clip: None,
            backward_clip: None,
            left_clip: None,
            right_clip: None,
            forward_left_clip: None,
            forward_right_clip: None,
            backward_left_clip: None,
            backward_right_clip: None,
            max_speed: max_speed.max(0.01),
        }
    }

    /// Assign an idle clip (placed at the origin).
    pub fn set_idle(&mut self, clip_index: usize) {
        self.idle_clip = Some(clip_index);
    }

    /// Assign a forward movement clip.
    pub fn set_forward(&mut self, clip_index: usize) {
        self.forward_clip = Some(clip_index);
    }

    /// Assign a backward movement clip.
    pub fn set_backward(&mut self, clip_index: usize) {
        self.backward_clip = Some(clip_index);
    }

    /// Assign a strafe-left clip.
    pub fn set_left(&mut self, clip_index: usize) {
        self.left_clip = Some(clip_index);
    }

    /// Assign a strafe-right clip.
    pub fn set_right(&mut self, clip_index: usize) {
        self.right_clip = Some(clip_index);
    }

    /// Assign a forward-left diagonal clip.
    pub fn set_forward_left(&mut self, clip_index: usize) {
        self.forward_left_clip = Some(clip_index);
    }

    /// Assign a forward-right diagonal clip.
    pub fn set_forward_right(&mut self, clip_index: usize) {
        self.forward_right_clip = Some(clip_index);
    }

    /// Assign a backward-left diagonal clip.
    pub fn set_backward_left(&mut self, clip_index: usize) {
        self.backward_left_clip = Some(clip_index);
    }

    /// Assign a backward-right diagonal clip.
    pub fn set_backward_right(&mut self, clip_index: usize) {
        self.backward_right_clip = Some(clip_index);
    }

    /// Build (or rebuild) the internal 2D blend space from the assigned clips.
    ///
    /// Call this after setting all clip slots. Points are placed at the
    /// canonical positions:
    /// - Idle: (0, 0)
    /// - Forward: (0, max_speed)
    /// - Backward: (0, -max_speed)
    /// - Left: (-max_speed, 0)
    /// - Right: (max_speed, 0)
    /// - Diagonals: at 45-degree offsets scaled by max_speed.
    pub fn build(&mut self) {
        self.inner.points.clear();

        let s = self.max_speed;
        let diag = s * std::f32::consts::FRAC_1_SQRT_2;

        if let Some(idx) = self.idle_clip {
            self.inner.add_point(Vec2::ZERO, idx);
        }
        if let Some(idx) = self.forward_clip {
            self.inner.add_point(Vec2::new(0.0, s), idx);
        }
        if let Some(idx) = self.backward_clip {
            self.inner.add_point(Vec2::new(0.0, -s), idx);
        }
        if let Some(idx) = self.left_clip {
            self.inner.add_point(Vec2::new(-s, 0.0), idx);
        }
        if let Some(idx) = self.right_clip {
            self.inner.add_point(Vec2::new(s, 0.0), idx);
        }
        if let Some(idx) = self.forward_left_clip {
            self.inner.add_point(Vec2::new(-diag, diag), idx);
        }
        if let Some(idx) = self.forward_right_clip {
            self.inner.add_point(Vec2::new(diag, diag), idx);
        }
        if let Some(idx) = self.backward_left_clip {
            self.inner.add_point(Vec2::new(-diag, -diag), idx);
        }
        if let Some(idx) = self.backward_right_clip {
            self.inner.add_point(Vec2::new(diag, -diag), idx);
        }

        self.inner.rebuild_triangulation();
    }

    /// Set the movement parameters from direction angle (radians, 0 = forward)
    /// and speed.
    pub fn set_direction_and_speed(&mut self, direction_rad: f32, speed: f32) {
        let clamped_speed = speed.clamp(0.0, self.max_speed);
        let x = direction_rad.sin() * clamped_speed;
        let y = direction_rad.cos() * clamped_speed;
        self.inner.set_parameter(Vec2::new(x, y));
    }

    /// Set the movement parameters from a velocity vector.
    pub fn set_velocity(&mut self, velocity: Vec2) {
        self.inner.set_parameter(velocity);
    }

    /// Evaluate and produce a blended pose.
    pub fn evaluate(
        &mut self,
        clips: &[AnimationClip],
        dt: f32,
        bone_count: usize,
    ) -> Vec<Transform> {
        self.inner.evaluate(clips, dt, bone_count)
    }

    /// Get the current blend weights.
    pub fn compute_weights(&mut self) -> Vec<(usize, f32)> {
        self.inner.compute_weights()
    }
}

// ---------------------------------------------------------------------------
// Delaunay triangulation (Bowyer-Watson algorithm)
// ---------------------------------------------------------------------------

/// Compute the Delaunay triangulation of a set of 2D blend points.
///
/// Uses the incremental Bowyer-Watson algorithm:
/// 1. Create a super-triangle that contains all points.
/// 2. Insert each point one at a time:
///    a. Find all triangles whose circumcircle contains the new point.
///    b. Remove those triangles, forming a star-shaped polygonal hole.
///    c. Re-triangulate the hole by connecting the new point to each edge.
/// 3. Remove triangles that share vertices with the super-triangle.
fn delaunay_triangulate(points: &[BlendPoint2D]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }

    // Find the bounding box.
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for pt in points {
        min_x = min_x.min(pt.position.x);
        min_y = min_y.min(pt.position.y);
        max_x = max_x.max(pt.position.x);
        max_y = max_y.max(pt.position.y);
    }

    let dx = (max_x - min_x).max(1.0);
    let dy = (max_y - min_y).max(1.0);
    let delta_max = dx.max(dy);
    let mid_x = (min_x + max_x) * 0.5;
    let mid_y = (min_y + max_y) * 0.5;

    // Super-triangle vertices (indices n, n+1, n+2).
    let margin = 20.0;
    let super_verts = [
        Vec2::new(mid_x - margin * delta_max, mid_y - margin * delta_max),
        Vec2::new(mid_x, mid_y + margin * delta_max),
        Vec2::new(mid_x + margin * delta_max, mid_y - margin * delta_max),
    ];

    // All vertex positions (original points + super triangle).
    let all_positions: Vec<Vec2> = points
        .iter()
        .map(|p| p.position)
        .chain(super_verts.iter().copied())
        .collect();

    let st_a = n;
    let st_b = n + 1;
    let st_c = n + 2;

    // Start with just the super-triangle.
    let mut triangles: Vec<[usize; 3]> = vec![[st_a, st_b, st_c]];

    // Insert each point.
    for i in 0..n {
        let p = all_positions[i];

        // Find "bad" triangles whose circumcircle contains p.
        let mut bad = Vec::new();
        for (t_idx, tri) in triangles.iter().enumerate() {
            let a = all_positions[tri[0]];
            let b = all_positions[tri[1]];
            let c = all_positions[tri[2]];
            if in_circumcircle(p, a, b, c) {
                bad.push(t_idx);
            }
        }

        // Collect the boundary edges of the polygonal hole.
        let mut boundary: Vec<[usize; 2]> = Vec::new();
        for &t_idx in &bad {
            let tri = triangles[t_idx];
            for edge_idx in 0..3 {
                let e = [tri[edge_idx], tri[(edge_idx + 1) % 3]];

                // An edge is on the boundary if it is NOT shared by another bad
                // triangle.
                let shared = bad.iter().any(|&other| {
                    other != t_idx && triangle_has_edge(triangles[other], e)
                });
                if !shared {
                    boundary.push(e);
                }
            }
        }

        // Remove bad triangles (in reverse order to preserve indices).
        bad.sort_unstable();
        for &t_idx in bad.iter().rev() {
            triangles.swap_remove(t_idx);
        }

        // Create new triangles from boundary edges to the new point.
        for edge in &boundary {
            triangles.push([i, edge[0], edge[1]]);
        }
    }

    // Remove any triangle that uses a super-triangle vertex.
    triangles.retain(|tri| {
        tri[0] < n && tri[1] < n && tri[2] < n
    });

    triangles
}

/// Check whether point `p` lies inside the circumcircle of triangle `(a, b, c)`.
fn in_circumcircle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    // Use the determinant method. The point is inside the circumcircle if
    // the determinant of the following matrix is positive (assuming CCW order):
    //
    //  | ax-px  ay-py  (ax-px)^2+(ay-py)^2 |
    //  | bx-px  by-py  (bx-px)^2+(by-py)^2 |
    //  | cx-px  cy-py  (cx-px)^2+(cy-py)^2 |
    //
    // We need to handle both CW and CCW orderings, so we use the absolute value
    // and check against the sign of the triangle's signed area.
    let d = a - p;
    let e = b - p;
    let f = c - p;

    let det = d.x * (e.y * (f.x * f.x + f.y * f.y) - f.y * (e.x * e.x + e.y * e.y))
        - d.y * (e.x * (f.x * f.x + f.y * f.y) - f.x * (e.x * e.x + e.y * e.y))
        + (d.x * d.x + d.y * d.y) * (e.x * f.y - e.y * f.x);

    // The sign depends on orientation. If triangle (a,b,c) is CCW, det > 0
    // means p is inside. If CW, det < 0 means inside. We check the triangle
    // orientation to determine which sign to test.
    let area_sign = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    if area_sign > 0.0 {
        det > 0.0
    } else {
        det < 0.0
    }
}

/// Check if a triangle contains an edge (in either direction).
fn triangle_has_edge(tri: [usize; 3], edge: [usize; 2]) -> bool {
    for i in 0..3 {
        let a = tri[i];
        let b = tri[(i + 1) % 3];
        if (a == edge[0] && b == edge[1]) || (a == edge[1] && b == edge[0]) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Barycentric coordinates
// ---------------------------------------------------------------------------

/// Compute barycentric coordinates of point `p` with respect to triangle
/// `(a, b, c)`.
///
/// Returns `None` if the triangle is degenerate (zero area). Otherwise returns
/// `(u, v, w)` where `u + v + w == 1`. Point `p` is inside the triangle iff
/// all three are non-negative.
fn barycentric(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> Option<(f32, f32, f32)> {
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;

    let dot00 = v0.dot(v0);
    let dot01 = v0.dot(v1);
    let dot02 = v0.dot(v2);
    let dot11 = v1.dot(v1);
    let dot12 = v1.dot(v2);

    let inv_denom = dot00 * dot11 - dot01 * dot01;
    if inv_denom.abs() < 1e-10 {
        return None; // Degenerate triangle.
    }
    let inv = 1.0 / inv_denom;

    let u = (dot11 * dot02 - dot01 * dot12) * inv; // weight for c
    let v = (dot00 * dot12 - dot01 * dot02) * inv; // weight for b
    let w = 1.0 - u - v; // weight for a

    Some((w, v, u)) // (weight_a, weight_b, weight_c)
}

/// Project point `p` onto the line segment `(a, b)` and return the parameter
/// `t` in [0, 1] and the distance from `p` to the projection.
fn project_onto_segment(p: Vec2, a: Vec2, b: Vec2) -> (f32, f32) {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-10 {
        return (0.0, (p - a).length());
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    let proj = a + ab * t;
    (t, (p - proj).length())
}

// ---------------------------------------------------------------------------
// Pose blending utility
// ---------------------------------------------------------------------------

/// Blend two poses by weight `t` (0 = pose `a`, 1 = pose `b`).
fn blend_poses(a: &[Transform], b: &[Transform], t: f32) -> Vec<Transform> {
    let t = t.clamp(0.0, 1.0);
    a.iter()
        .zip(b.iter())
        .map(|(ta, tb)| ta.lerp(tb, t))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{AnimationClip, BoneTrack, Keyframe};
    use glam::{Quat, Vec2, Vec3};

    fn make_clip(name: &str, bone_count: usize) -> AnimationClip {
        let mut clip = AnimationClip::new(name, 1.0);
        clip.looping = true;
        for i in 0..bone_count {
            let mut track = BoneTrack::new(i);
            track.position_keys = vec![
                Keyframe::new(0.0, Vec3::new(0.0, i as f32, 0.0)),
                Keyframe::new(1.0, Vec3::new(0.0, i as f32, 0.0)),
            ];
            track.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
            track.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
            clip.add_track(track);
        }
        clip
    }

    // -- BlendSpace1D tests --

    #[test]
    fn test_1d_empty() {
        let bs = BlendSpace1D::new("empty");
        assert!(bs.compute_weights().is_empty());
    }

    #[test]
    fn test_1d_single_point() {
        let mut bs = BlendSpace1D::new("single");
        bs.add_point(0.0, 0);
        let w = bs.compute_weights();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0], (0, 1.0));
    }

    #[test]
    fn test_1d_two_points_at_extremes() {
        let mut bs = BlendSpace1D::new("test");
        bs.add_point(0.0, 0);
        bs.add_point(1.0, 1);

        bs.set_parameter(0.0);
        let w = bs.compute_weights();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0], (0, 1.0));

        bs.set_parameter(1.0);
        let w = bs.compute_weights();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0], (1, 1.0));
    }

    #[test]
    fn test_1d_two_points_midpoint() {
        let mut bs = BlendSpace1D::new("test");
        bs.add_point(0.0, 0);
        bs.add_point(1.0, 1);

        bs.set_parameter(0.5);
        let w = bs.compute_weights();
        assert_eq!(w.len(), 2);
        assert!((w[0].1 - 0.5).abs() < 0.01);
        assert!((w[1].1 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_1d_three_points() {
        let mut bs = BlendSpace1D::new("test");
        bs.add_point(0.0, 0); // idle
        bs.add_point(5.0, 1); // walk
        bs.add_point(10.0, 2); // run

        bs.set_parameter(2.5);
        let w = bs.compute_weights();
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].0, 0); // idle
        assert_eq!(w[1].0, 1); // walk
        assert!((w[0].1 - 0.5).abs() < 0.01);
        assert!((w[1].1 - 0.5).abs() < 0.01);

        bs.set_parameter(7.5);
        let w = bs.compute_weights();
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].0, 1); // walk
        assert_eq!(w[1].0, 2); // run
    }

    #[test]
    fn test_1d_below_range() {
        let mut bs = BlendSpace1D::new("test");
        bs.add_point(1.0, 0);
        bs.add_point(2.0, 1);

        bs.set_parameter(0.0);
        let w = bs.compute_weights();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].0, 0);
    }

    #[test]
    fn test_1d_above_range() {
        let mut bs = BlendSpace1D::new("test");
        bs.add_point(1.0, 0);
        bs.add_point(2.0, 1);

        bs.set_parameter(5.0);
        let w = bs.compute_weights();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].0, 1);
    }

    #[test]
    fn test_1d_evaluate() {
        let clips = vec![make_clip("idle", 2), make_clip("walk", 2)];
        let mut bs = BlendSpace1D::new("locomotion");
        bs.add_point(0.0, 0);
        bs.add_point(1.0, 1);
        bs.set_parameter(0.5);

        let pose = bs.evaluate(&clips, 0.016, 2);
        assert_eq!(pose.len(), 2);
    }

    // -- Delaunay triangulation tests --

    #[test]
    fn test_delaunay_three_points() {
        let points = vec![
            BlendPoint2D { position: Vec2::new(0.0, 0.0), clip_index: 0, speed: 1.0 },
            BlendPoint2D { position: Vec2::new(1.0, 0.0), clip_index: 1, speed: 1.0 },
            BlendPoint2D { position: Vec2::new(0.5, 1.0), clip_index: 2, speed: 1.0 },
        ];
        let tris = delaunay_triangulate(&points);
        assert_eq!(tris.len(), 1);
        // All three point indices should be present.
        let mut indices: Vec<usize> = tris[0].to_vec();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_delaunay_four_points() {
        let points = vec![
            BlendPoint2D { position: Vec2::new(0.0, 0.0), clip_index: 0, speed: 1.0 },
            BlendPoint2D { position: Vec2::new(2.0, 0.0), clip_index: 1, speed: 1.0 },
            BlendPoint2D { position: Vec2::new(2.0, 2.0), clip_index: 2, speed: 1.0 },
            BlendPoint2D { position: Vec2::new(0.0, 2.0), clip_index: 3, speed: 1.0 },
        ];
        let tris = delaunay_triangulate(&points);
        assert_eq!(tris.len(), 2); // Square -> 2 triangles.
    }

    #[test]
    fn test_delaunay_too_few_points() {
        let points = vec![
            BlendPoint2D { position: Vec2::new(0.0, 0.0), clip_index: 0, speed: 1.0 },
            BlendPoint2D { position: Vec2::new(1.0, 0.0), clip_index: 1, speed: 1.0 },
        ];
        let tris = delaunay_triangulate(&points);
        assert!(tris.is_empty());
    }

    #[test]
    fn test_delaunay_five_points() {
        // Diamond + center.
        let points = vec![
            BlendPoint2D { position: Vec2::new(0.0, 0.0), clip_index: 0, speed: 1.0 },
            BlendPoint2D { position: Vec2::new(1.0, 0.0), clip_index: 1, speed: 1.0 },
            BlendPoint2D { position: Vec2::new(0.0, 1.0), clip_index: 2, speed: 1.0 },
            BlendPoint2D { position: Vec2::new(1.0, 1.0), clip_index: 3, speed: 1.0 },
            BlendPoint2D { position: Vec2::new(0.5, 0.5), clip_index: 4, speed: 1.0 },
        ];
        let tris = delaunay_triangulate(&points);
        assert!(tris.len() >= 4, "Expected at least 4 triangles, got {}", tris.len());
    }

    // -- Barycentric tests --

    #[test]
    fn test_barycentric_center() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 0.0);
        let c = Vec2::new(0.0, 3.0);
        let center = Vec2::new(1.0, 1.0);

        let (u, v, w) = barycentric(center, a, b, c).unwrap();
        assert!((u + v + w - 1.0).abs() < 0.01);
        assert!(u > 0.0 && v > 0.0 && w > 0.0);
    }

    #[test]
    fn test_barycentric_at_vertex() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(0.0, 1.0);

        let (u, v, w) = barycentric(a, a, b, c).unwrap();
        assert!((u - 1.0).abs() < 0.01);
        assert!(v.abs() < 0.01);
        assert!(w.abs() < 0.01);
    }

    #[test]
    fn test_barycentric_on_edge() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 0.0);
        let c = Vec2::new(0.0, 2.0);
        let mid_ab = Vec2::new(1.0, 0.0);

        let (u, v, w) = barycentric(mid_ab, a, b, c).unwrap();
        assert!(w.abs() < 0.01); // No weight on c.
        assert!((u + v - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_barycentric_outside() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(0.0, 1.0);
        let outside = Vec2::new(-1.0, -1.0);

        let (u, v, w) = barycentric(outside, a, b, c).unwrap();
        // At least one coordinate should be negative.
        assert!(u < 0.0 || v < 0.0 || w < 0.0);
    }

    // -- BlendSpace2D tests --

    #[test]
    fn test_2d_empty() {
        let mut bs = BlendSpace2D::new("empty");
        assert!(bs.compute_weights().is_empty());
    }

    #[test]
    fn test_2d_single_point() {
        let mut bs = BlendSpace2D::new("single");
        bs.add_point(Vec2::ZERO, 0);
        let w = bs.compute_weights();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].0, 0);
    }

    #[test]
    fn test_2d_two_points() {
        let mut bs = BlendSpace2D::new("two");
        bs.add_point(Vec2::new(0.0, 0.0), 0);
        bs.add_point(Vec2::new(1.0, 0.0), 1);

        bs.set_parameter(Vec2::new(0.5, 0.0));
        let w = bs.compute_weights();
        assert_eq!(w.len(), 2);
        assert!((w[0].1 - 0.5).abs() < 0.01);
        assert!((w[1].1 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_2d_triangle_inside() {
        let mut bs = BlendSpace2D::new("tri");
        bs.add_point(Vec2::new(0.0, 0.0), 0);
        bs.add_point(Vec2::new(2.0, 0.0), 1);
        bs.add_point(Vec2::new(1.0, 2.0), 2);

        bs.set_parameter(Vec2::new(1.0, 0.5));
        let w = bs.compute_weights();
        assert!(!w.is_empty());
        let total: f32 = w.iter().map(|(_, wt)| *wt).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "Weights should sum to 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_2d_at_vertex() {
        let mut bs = BlendSpace2D::new("tri");
        bs.add_point(Vec2::new(0.0, 0.0), 0);
        bs.add_point(Vec2::new(1.0, 0.0), 1);
        bs.add_point(Vec2::new(0.0, 1.0), 2);

        bs.set_parameter(Vec2::new(0.0, 0.0));
        let w = bs.compute_weights();
        // Should be mostly clip 0.
        let clip0_weight: f32 = w
            .iter()
            .filter(|&&(idx, _)| idx == 0)
            .map(|(_, w)| *w)
            .sum();
        assert!(clip0_weight > 0.9, "Expected mostly clip 0, got {}", clip0_weight);
    }

    #[test]
    fn test_2d_outside_hull() {
        let mut bs = BlendSpace2D::new("tri");
        bs.add_point(Vec2::new(0.0, 0.0), 0);
        bs.add_point(Vec2::new(1.0, 0.0), 1);
        bs.add_point(Vec2::new(0.5, 1.0), 2);

        // Way outside.
        bs.set_parameter(Vec2::new(5.0, 5.0));
        let w = bs.compute_weights();
        // Should still return something reasonable (nearest edge or point).
        assert!(!w.is_empty());
    }

    #[test]
    fn test_2d_evaluate() {
        let clips = vec![make_clip("a", 2), make_clip("b", 2), make_clip("c", 2)];
        let mut bs = BlendSpace2D::new("test");
        bs.add_point(Vec2::new(0.0, 0.0), 0);
        bs.add_point(Vec2::new(1.0, 0.0), 1);
        bs.add_point(Vec2::new(0.5, 1.0), 2);

        bs.set_parameter(Vec2::new(0.5, 0.3));
        let pose = bs.evaluate(&clips, 0.016, 2);
        assert_eq!(pose.len(), 2);
    }

    // -- DirectionalBlendSpace tests --

    #[test]
    fn test_directional_build() {
        let mut dbs = DirectionalBlendSpace::new("locomotion", 5.0);
        dbs.set_idle(0);
        dbs.set_forward(1);
        dbs.set_backward(2);
        dbs.set_left(3);
        dbs.set_right(4);
        dbs.build();

        assert_eq!(dbs.inner.point_count(), 5);
        assert!(dbs.inner.triangle_count() >= 2);
    }

    #[test]
    fn test_directional_idle() {
        let mut dbs = DirectionalBlendSpace::new("locomotion", 5.0);
        dbs.set_idle(0);
        dbs.set_forward(1);
        dbs.set_backward(2);
        dbs.set_left(3);
        dbs.set_right(4);
        dbs.build();

        // Speed 0 -> should be fully idle.
        dbs.set_direction_and_speed(0.0, 0.0);
        let w = dbs.compute_weights();
        let idle_w: f32 = w
            .iter()
            .filter(|&&(idx, _)| idx == 0)
            .map(|(_, w)| *w)
            .sum();
        assert!(
            idle_w > 0.5,
            "At zero speed, idle weight should be dominant: {}",
            idle_w
        );
    }

    #[test]
    fn test_directional_forward() {
        let mut dbs = DirectionalBlendSpace::new("locomotion", 5.0);
        dbs.set_idle(0);
        dbs.set_forward(1);
        dbs.set_backward(2);
        dbs.set_left(3);
        dbs.set_right(4);
        dbs.build();

        // Full speed forward.
        dbs.set_direction_and_speed(0.0, 5.0);
        let w = dbs.compute_weights();
        let fwd_w: f32 = w
            .iter()
            .filter(|&&(idx, _)| idx == 1)
            .map(|(_, w)| *w)
            .sum();
        assert!(
            fwd_w > 0.5,
            "At full forward speed, forward weight should be dominant: {}",
            fwd_w
        );
    }

    #[test]
    fn test_directional_with_diagonals() {
        let mut dbs = DirectionalBlendSpace::new("locomotion", 5.0);
        dbs.set_idle(0);
        dbs.set_forward(1);
        dbs.set_backward(2);
        dbs.set_left(3);
        dbs.set_right(4);
        dbs.set_forward_left(5);
        dbs.set_forward_right(6);
        dbs.set_backward_left(7);
        dbs.set_backward_right(8);
        dbs.build();

        assert_eq!(dbs.inner.point_count(), 9);
        assert!(dbs.inner.triangle_count() >= 8);
    }

    #[test]
    fn test_directional_velocity() {
        let mut dbs = DirectionalBlendSpace::new("test", 5.0);
        dbs.set_idle(0);
        dbs.set_forward(1);
        dbs.set_right(2);
        dbs.build();

        dbs.set_velocity(Vec2::new(2.5, 2.5));
        let w = dbs.compute_weights();
        assert!(!w.is_empty());
    }

    // -- project_onto_segment tests --

    #[test]
    fn test_project_onto_segment_midpoint() {
        let (t, dist) = project_onto_segment(
            Vec2::new(0.5, 1.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
        );
        assert!((t - 0.5).abs() < 0.01);
        assert!((dist - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_project_onto_segment_clamp() {
        let (t, _) = project_onto_segment(
            Vec2::new(-5.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
        );
        assert!((t).abs() < 0.01); // Clamped to 0.

        let (t, _) = project_onto_segment(
            Vec2::new(5.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
        );
        assert!((t - 1.0).abs() < 0.01); // Clamped to 1.
    }

    // -- in_circumcircle tests --

    #[test]
    fn test_in_circumcircle_inside() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 0.0);
        let c = Vec2::new(1.0, 2.0);
        let p = Vec2::new(1.0, 0.5);
        assert!(in_circumcircle(p, a, b, c));
    }

    #[test]
    fn test_in_circumcircle_outside() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(0.5, 0.5);
        let p = Vec2::new(10.0, 10.0);
        assert!(!in_circumcircle(p, a, b, c));
    }
}
