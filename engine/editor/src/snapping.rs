//! Grid and surface snapping for the editor.
//!
//! Provides snapping utilities for positioning, rotating, and scaling entities
//! to discrete grid values, nearby surfaces, vertices, and edges.
//!
//! # Features
//!
//! - **Grid snapping** — snap positions to a configurable grid.
//! - **Rotation snapping** — snap rotations to discrete angle increments.
//! - **Scale snapping** — snap scale values to discrete increments.
//! - **Surface snapping** — raycast downward to snap to the nearest surface.
//! - **Vertex snapping** — snap to nearby mesh vertices.
//! - **Edge snapping** — snap to nearby mesh edges.
//! - **Snap guides** — visual guide lines showing alignment with other entities.
//!
//! # Math
//!
//! Grid snapping uses `round(value / grid_size) * grid_size` for each axis
//! independently. Rotation snapping converts the rotation angle to degrees,
//! rounds to the nearest snap increment, and converts back. Scale snapping
//! works similarly to grid snapping but on scale values.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Vec3 — local 3D vector for self-contained snapping math
// ---------------------------------------------------------------------------

/// 3D vector for snapping operations.
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
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Squared distance to another point.
    pub fn distance_sq(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Distance to another point.
    pub fn distance(&self, other: &Self) -> f32 {
        self.distance_sq(other).sqrt()
    }

    /// Squared length.
    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Length.
    pub fn length(&self) -> f32 {
        self.length_sq().sqrt()
    }

    /// Dot product.
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Subtract.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Add.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Scale by a scalar.
    pub fn scale(&self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Lerp between self and other.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

// ---------------------------------------------------------------------------
// Quat — local quaternion for rotation snapping
// ---------------------------------------------------------------------------

/// Quaternion for rotation snapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Create from axis-angle (angle in radians).
    pub fn from_axis_angle(axis: &Vec3, angle: f32) -> Self {
        let half = angle * 0.5;
        let s = half.sin();
        let len = axis.length();
        if len < 1e-6 {
            return Self::IDENTITY;
        }
        let inv_len = 1.0 / len;
        Self {
            x: axis.x * inv_len * s,
            y: axis.y * inv_len * s,
            z: axis.z * inv_len * s,
            w: half.cos(),
        }
    }

    /// Convert to axis and angle (angle in radians).
    pub fn to_axis_angle(&self) -> (Vec3, f32) {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len < 1e-6 {
            return (Vec3::new(0.0, 1.0, 0.0), 0.0);
        }
        let inv_len = 1.0 / len;
        let angle = 2.0 * len.atan2(self.w);
        let axis = Vec3::new(self.x * inv_len, self.y * inv_len, self.z * inv_len);
        (axis, angle)
    }

    /// Normalize the quaternion.
    pub fn normalize(&self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if len < 1e-6 {
            return Self::IDENTITY;
        }
        let inv = 1.0 / len;
        Self {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
            w: self.w * inv,
        }
    }

    /// Convert to Euler angles (pitch, yaw, roll) in radians.
    pub fn to_euler(&self) -> (f32, f32, f32) {
        let sinp = 2.0 * (self.w * self.x - self.y * self.z);
        let pitch = if sinp.abs() >= 1.0 {
            std::f32::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (self.w * self.y + self.z * self.x);
        let cosy_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y);
        let yaw = siny_cosp.atan2(cosy_cosp);

        let sinr_cosp = 2.0 * (self.w * self.z + self.x * self.y);
        let cosr_cosp = 1.0 - 2.0 * (self.z * self.z + self.x * self.x);
        let roll = sinr_cosp.atan2(cosr_cosp);

        (pitch, yaw, roll)
    }

    /// Create from Euler angles (pitch, yaw, roll) in radians.
    pub fn from_euler(pitch: f32, yaw: f32, roll: f32) -> Self {
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sy, cy) = (yaw * 0.5).sin_cos();
        let (sr, cr) = (roll * 0.5).sin_cos();

        Self {
            x: cp * sy * sr + sp * cy * cr,
            y: cp * sy * cr - sp * cy * sr,
            z: cp * cy * sr - sp * sy * cr,
            w: cp * cy * cr + sp * sy * sr,
        }
    }
}

// ---------------------------------------------------------------------------
// SnapSettings
// ---------------------------------------------------------------------------

/// Configuration for all snapping behaviors.
#[derive(Debug, Clone)]
pub struct SnapSettings {
    /// Whether grid snapping is enabled.
    pub enabled: bool,
    /// Grid cell size for position snapping.
    pub grid_size: f32,
    /// Rotation snap increment in degrees.
    pub rotation_snap: f32,
    /// Scale snap increment.
    pub scale_snap: f32,
    /// Whether surface snapping is enabled.
    pub surface_snap_enabled: bool,
    /// Whether vertex snapping is enabled.
    pub vertex_snap_enabled: bool,
    /// Vertex snap threshold (max distance to snap to a vertex).
    pub vertex_snap_threshold: f32,
    /// Whether edge snapping is enabled.
    pub edge_snap_enabled: bool,
    /// Edge snap threshold (max distance to snap to an edge).
    pub edge_snap_threshold: f32,
    /// Whether to show snap guide lines.
    pub show_guides: bool,
    /// Guide line search distance.
    pub guide_search_distance: f32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            grid_size: 1.0,
            rotation_snap: 15.0,
            scale_snap: 0.1,
            surface_snap_enabled: false,
            vertex_snap_enabled: false,
            vertex_snap_threshold: 0.5,
            edge_snap_enabled: false,
            edge_snap_threshold: 0.5,
            show_guides: true,
            guide_search_distance: 50.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Grid snapping
// ---------------------------------------------------------------------------

/// Snap a position to the nearest grid point.
///
/// Each axis is snapped independently:
/// `snapped = round(pos / grid_size) * grid_size`
pub fn snap_to_grid(position: Vec3, grid_size: f32) -> Vec3 {
    if grid_size <= 0.0 {
        return position;
    }

    Vec3::new(
        (position.x / grid_size).round() * grid_size,
        (position.y / grid_size).round() * grid_size,
        (position.z / grid_size).round() * grid_size,
    )
}

/// Snap a single axis value to the grid.
pub fn snap_value_to_grid(value: f32, grid_size: f32) -> f32 {
    if grid_size <= 0.0 {
        return value;
    }
    (value / grid_size).round() * grid_size
}

/// Snap position with per-axis grid sizes.
pub fn snap_to_grid_per_axis(position: Vec3, grid_x: f32, grid_y: f32, grid_z: f32) -> Vec3 {
    Vec3::new(
        snap_value_to_grid(position.x, grid_x),
        snap_value_to_grid(position.y, grid_y),
        snap_value_to_grid(position.z, grid_z),
    )
}

// ---------------------------------------------------------------------------
// Rotation snapping
// ---------------------------------------------------------------------------

/// Snap a rotation quaternion so that its Euler angles are multiples of
/// `snap_degrees`.
///
/// The algorithm:
/// 1. Convert quaternion to Euler angles (pitch, yaw, roll).
/// 2. Snap each angle to the nearest multiple of `snap_degrees`.
/// 3. Convert back to a quaternion.
pub fn snap_rotation(rotation: Quat, snap_degrees: f32) -> Quat {
    if snap_degrees <= 0.0 {
        return rotation;
    }

    let snap_rad = snap_degrees * PI / 180.0;
    let (pitch, yaw, roll) = rotation.to_euler();

    let snapped_pitch = (pitch / snap_rad).round() * snap_rad;
    let snapped_yaw = (yaw / snap_rad).round() * snap_rad;
    let snapped_roll = (roll / snap_rad).round() * snap_rad;

    Quat::from_euler(snapped_pitch, snapped_yaw, snapped_roll).normalize()
}

/// Snap a single angle (in degrees) to the nearest snap increment.
pub fn snap_angle_degrees(angle: f32, snap_degrees: f32) -> f32 {
    if snap_degrees <= 0.0 {
        return angle;
    }
    (angle / snap_degrees).round() * snap_degrees
}

/// Snap a single angle (in radians) to the nearest snap increment (given
/// in degrees).
pub fn snap_angle_radians(angle_rad: f32, snap_degrees: f32) -> f32 {
    if snap_degrees <= 0.0 {
        return angle_rad;
    }
    let snap_rad = snap_degrees * PI / 180.0;
    (angle_rad / snap_rad).round() * snap_rad
}

// ---------------------------------------------------------------------------
// Scale snapping
// ---------------------------------------------------------------------------

/// Snap a scale vector to the nearest snap increment.
pub fn snap_scale(scale: Vec3, snap_increment: f32) -> Vec3 {
    if snap_increment <= 0.0 {
        return scale;
    }

    Vec3::new(
        (scale.x / snap_increment).round() * snap_increment,
        (scale.y / snap_increment).round() * snap_increment,
        (scale.z / snap_increment).round() * snap_increment,
    )
}

/// Snap a uniform scale value.
pub fn snap_uniform_scale(scale: f32, snap_increment: f32) -> f32 {
    if snap_increment <= 0.0 {
        return scale;
    }
    (scale / snap_increment).round() * snap_increment
}

// ---------------------------------------------------------------------------
// Surface snapping
// ---------------------------------------------------------------------------

/// A triangle in world space, used for surface raycasting.
#[derive(Debug, Clone)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
}

impl Triangle {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        Self { v0, v1, v2 }
    }

    /// Compute the triangle normal (not necessarily normalized).
    pub fn normal(&self) -> Vec3 {
        let e1 = self.v1.sub(&self.v0);
        let e2 = self.v2.sub(&self.v0);
        Vec3::new(
            e1.y * e2.z - e1.z * e2.y,
            e1.z * e2.x - e1.x * e2.z,
            e1.x * e2.y - e1.y * e2.x,
        )
    }
}

/// Result of a surface snap raycast.
#[derive(Debug, Clone)]
pub struct SurfaceHit {
    pub position: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

/// Snap a position to the nearest surface below it by raycasting downward.
///
/// `triangles` is a list of scene mesh triangles to test against.
/// Returns the hit position and normal, or `None` if no surface is found.
///
/// Uses Moller-Trumbore ray-triangle intersection.
pub fn snap_to_surface(position: Vec3, triangles: &[Triangle]) -> Option<SurfaceHit> {
    let ray_origin = position;
    let ray_dir = Vec3::new(0.0, -1.0, 0.0); // Cast downward.

    let mut closest: Option<SurfaceHit> = None;

    for tri in triangles {
        if let Some(t) = ray_triangle_intersect(&ray_origin, &ray_dir, tri) {
            if t >= 0.0 {
                let hit_pos = ray_origin.add(&ray_dir.scale(t));
                let normal = tri.normal();
                let n_len = normal.length();
                let normalized_normal = if n_len > 1e-6 {
                    normal.scale(1.0 / n_len)
                } else {
                    Vec3::new(0.0, 1.0, 0.0)
                };

                let hit = SurfaceHit {
                    position: hit_pos,
                    normal: normalized_normal,
                    distance: t,
                };

                if closest
                    .as_ref()
                    .map_or(true, |c| hit.distance < c.distance)
                {
                    closest = Some(hit);
                }
            }
        }
    }

    closest
}

/// Moller-Trumbore ray-triangle intersection.
/// Returns the parametric t along the ray, or None if no intersection.
fn ray_triangle_intersect(origin: &Vec3, dir: &Vec3, tri: &Triangle) -> Option<f32> {
    let epsilon = 1e-6;

    let e1 = tri.v1.sub(&tri.v0);
    let e2 = tri.v2.sub(&tri.v0);

    // h = dir x e2
    let h = Vec3::new(
        dir.y * e2.z - dir.z * e2.y,
        dir.z * e2.x - dir.x * e2.z,
        dir.x * e2.y - dir.y * e2.x,
    );

    let a = e1.dot(&h);
    if a.abs() < epsilon {
        return None; // Ray parallel to triangle.
    }

    let f = 1.0 / a;
    let s = origin.sub(&tri.v0);
    let u = f * s.dot(&h);
    if u < 0.0 || u > 1.0 {
        return None;
    }

    // q = s x e1
    let q = Vec3::new(
        s.y * e1.z - s.z * e1.y,
        s.z * e1.x - s.x * e1.z,
        s.x * e1.y - s.y * e1.x,
    );

    let v = f * dir.dot(&q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * e2.dot(&q);
    if t > epsilon {
        Some(t)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Vertex snapping
// ---------------------------------------------------------------------------

/// Snap a position to the nearest vertex within the threshold distance.
///
/// Returns the snapped position if a vertex is within range, otherwise
/// returns the original position.
pub fn snap_to_vertex(position: Vec3, vertices: &[Vec3], threshold: f32) -> Vec3 {
    let threshold_sq = threshold * threshold;
    let mut closest_dist_sq = f32::MAX;
    let mut closest_vertex = position;

    for v in vertices {
        let dist_sq = position.distance_sq(v);
        if dist_sq < threshold_sq && dist_sq < closest_dist_sq {
            closest_dist_sq = dist_sq;
            closest_vertex = *v;
        }
    }

    closest_vertex
}

// ---------------------------------------------------------------------------
// Edge snapping
// ---------------------------------------------------------------------------

/// An edge defined by two endpoints.
#[derive(Debug, Clone)]
pub struct Edge {
    pub a: Vec3,
    pub b: Vec3,
}

impl Edge {
    pub fn new(a: Vec3, b: Vec3) -> Self {
        Self { a, b }
    }

    /// Length of the edge.
    pub fn length(&self) -> f32 {
        self.a.distance(&self.b)
    }

    /// Midpoint of the edge.
    pub fn midpoint(&self) -> Vec3 {
        self.a.lerp(&self.b, 0.5)
    }

    /// Find the closest point on this edge to a given point.
    ///
    /// Returns the closest point and the parametric t value (0 = at a, 1 = at b).
    pub fn closest_point(&self, point: &Vec3) -> (Vec3, f32) {
        let ab = self.b.sub(&self.a);
        let ap = point.sub(&self.a);

        let ab_len_sq = ab.length_sq();
        if ab_len_sq < 1e-10 {
            return (self.a, 0.0);
        }

        let t = ap.dot(&ab) / ab_len_sq;
        let t_clamped = t.clamp(0.0, 1.0);
        let closest = self.a.add(&ab.scale(t_clamped));
        (closest, t_clamped)
    }
}

/// Snap a position to the nearest edge within the threshold distance.
///
/// Returns the snapped position (the closest point on the nearest edge)
/// if an edge is within range, otherwise returns the original position.
pub fn snap_to_edge(position: Vec3, edges: &[Edge], threshold: f32) -> Vec3 {
    let threshold_sq = threshold * threshold;
    let mut closest_dist_sq = f32::MAX;
    let mut closest_point = position;

    for edge in edges {
        let (point, _t) = edge.closest_point(&position);
        let dist_sq = position.distance_sq(&point);

        if dist_sq < threshold_sq && dist_sq < closest_dist_sq {
            closest_dist_sq = dist_sq;
            closest_point = point;
        }
    }

    closest_point
}

// ---------------------------------------------------------------------------
// SnapGuide — visual alignment guide lines
// ---------------------------------------------------------------------------

/// A visual guide line shown when an entity aligns with other entities or
/// grid lines during a drag operation.
#[derive(Debug, Clone)]
pub struct SnapGuide {
    /// Guide type.
    pub guide_type: SnapGuideType,
    /// Start point of the guide line (world space).
    pub start: Vec3,
    /// End point of the guide line (world space).
    pub end: Vec3,
    /// Which axis this guide represents (0=X, 1=Y, 2=Z).
    pub axis: usize,
    /// Color of the guide line (RGBA).
    pub color: [f32; 4],
}

/// Type of snap guide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapGuideType {
    /// Alignment with another entity on an axis.
    EntityAlignment,
    /// Alignment with the grid.
    GridAlignment,
    /// Equal spacing between entities.
    EqualSpacing,
}

impl SnapGuide {
    /// Create an entity alignment guide.
    pub fn entity_alignment(axis: usize, start: Vec3, end: Vec3) -> Self {
        let color = match axis {
            0 => [1.0, 0.3, 0.3, 0.8], // Red for X.
            1 => [0.3, 1.0, 0.3, 0.8], // Green for Y.
            _ => [0.3, 0.3, 1.0, 0.8], // Blue for Z.
        };
        Self {
            guide_type: SnapGuideType::EntityAlignment,
            start,
            end,
            axis,
            color,
        }
    }

    /// Length of the guide line.
    pub fn length(&self) -> f32 {
        self.start.distance(&self.end)
    }
}

/// Find alignment guides for a position relative to a set of reference
/// entity positions.
///
/// Returns guides for each axis where the position aligns (within
/// `tolerance`) with any reference entity.
pub fn find_alignment_guides(
    position: Vec3,
    reference_positions: &[Vec3],
    tolerance: f32,
    guide_length: f32,
) -> Vec<SnapGuide> {
    let mut guides = Vec::new();

    for ref_pos in reference_positions {
        for axis in 0..3 {
            let pos_val = match axis {
                0 => position.x,
                1 => position.y,
                _ => position.z,
            };
            let ref_val = match axis {
                0 => ref_pos.x,
                1 => ref_pos.y,
                _ => ref_pos.z,
            };

            if (pos_val - ref_val).abs() < tolerance {
                let half = guide_length * 0.5;
                let (start, end) = match axis {
                    0 => (
                        Vec3::new(pos_val, position.y - half, position.z),
                        Vec3::new(pos_val, position.y + half, position.z),
                    ),
                    1 => (
                        Vec3::new(position.x - half, pos_val, position.z),
                        Vec3::new(position.x + half, pos_val, position.z),
                    ),
                    _ => (
                        Vec3::new(position.x, position.y, pos_val - half),
                        Vec3::new(position.x, position.y, pos_val + half),
                    ),
                };
                guides.push(SnapGuide::entity_alignment(axis, start, end));
            }
        }
    }

    guides
}

/// Apply all enabled snapping from a SnapSettings.
pub fn apply_snap(position: Vec3, settings: &SnapSettings) -> Vec3 {
    if !settings.enabled {
        return position;
    }
    snap_to_grid(position, settings.grid_size)
}

/// Apply rotation snapping from settings.
pub fn apply_rotation_snap(rotation: Quat, settings: &SnapSettings) -> Quat {
    if !settings.enabled || settings.rotation_snap <= 0.0 {
        return rotation;
    }
    snap_rotation(rotation, settings.rotation_snap)
}

/// Apply scale snapping from settings.
pub fn apply_scale_snap(scale: Vec3, settings: &SnapSettings) -> Vec3 {
    if !settings.enabled || settings.scale_snap <= 0.0 {
        return scale;
    }
    snap_scale(scale, settings.scale_snap)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Grid snapping tests ----------------------------------------------------

    #[test]
    fn snap_to_grid_basic() {
        let pos = Vec3::new(1.3, 2.7, -0.2);
        let snapped = snap_to_grid(pos, 1.0);
        assert!((snapped.x - 1.0).abs() < 1e-5);
        assert!((snapped.y - 3.0).abs() < 1e-5);
        assert!((snapped.z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn snap_to_grid_half_unit() {
        let pos = Vec3::new(1.3, 2.7, 0.1);
        let snapped = snap_to_grid(pos, 0.5);
        assert!((snapped.x - 1.5).abs() < 1e-5);
        assert!((snapped.y - 2.5).abs() < 1e-5);
        assert!((snapped.z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn snap_to_grid_exact() {
        let pos = Vec3::new(2.0, 4.0, 6.0);
        let snapped = snap_to_grid(pos, 2.0);
        assert_eq!(snapped, pos);
    }

    #[test]
    fn snap_to_grid_zero_size_noop() {
        let pos = Vec3::new(1.5, 2.5, 3.5);
        let snapped = snap_to_grid(pos, 0.0);
        assert_eq!(snapped, pos);
    }

    #[test]
    fn snap_to_grid_negative_values() {
        let pos = Vec3::new(-1.3, -2.7, -0.2);
        let snapped = snap_to_grid(pos, 1.0);
        assert!((snapped.x - (-1.0)).abs() < 1e-5);
        assert!((snapped.y - (-3.0)).abs() < 1e-5);
        assert!((snapped.z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn snap_value_to_grid_test() {
        assert!((snap_value_to_grid(1.3, 0.5) - 1.5).abs() < 1e-5);
        assert!((snap_value_to_grid(1.2, 0.5) - 1.0).abs() < 1e-5);
        assert!((snap_value_to_grid(1.25, 0.5) - 1.5).abs() < 1e-5);
    }

    #[test]
    fn snap_per_axis() {
        let pos = Vec3::new(1.3, 2.7, 3.1);
        let snapped = snap_to_grid_per_axis(pos, 1.0, 0.5, 0.25);
        assert!((snapped.x - 1.0).abs() < 1e-5);
        assert!((snapped.y - 2.5).abs() < 1e-5);
        assert!((snapped.z - 3.0).abs() < 1e-5);
    }

    // -- Rotation snapping tests ------------------------------------------------

    #[test]
    fn snap_angle_degrees_test() {
        assert!((snap_angle_degrees(37.0, 15.0) - 30.0).abs() < 1e-3);
        assert!((snap_angle_degrees(42.0, 15.0) - 45.0).abs() < 1e-3);
        assert!((snap_angle_degrees(90.0, 15.0) - 90.0).abs() < 1e-3);
    }

    #[test]
    fn snap_angle_degrees_zero_noop() {
        assert!((snap_angle_degrees(37.0, 0.0) - 37.0).abs() < 1e-5);
    }

    #[test]
    fn snap_angle_radians_test() {
        let angle = PI / 4.0; // 45 degrees.
        let snapped = snap_angle_radians(angle, 15.0);
        assert!((snapped - PI / 4.0).abs() < 1e-4);

        let angle2 = PI / 4.0 + 0.05; // ~47.9 degrees.
        let snapped2 = snap_angle_radians(angle2, 15.0);
        assert!((snapped2 - PI / 4.0).abs() < 1e-4); // Should snap to 45.
    }

    #[test]
    fn snap_rotation_identity() {
        let snapped = snap_rotation(Quat::IDENTITY, 15.0);
        // Identity should remain identity (or very close).
        assert!((snapped.w - 1.0).abs() < 1e-4);
    }

    // -- Scale snapping tests ---------------------------------------------------

    #[test]
    fn snap_scale_basic() {
        let scale = Vec3::new(1.23, 0.47, 2.91);
        let snapped = snap_scale(scale, 0.1);
        assert!((snapped.x - 1.2).abs() < 1e-5);
        assert!((snapped.y - 0.5).abs() < 1e-5);
        assert!((snapped.z - 2.9).abs() < 1e-5);
    }

    #[test]
    fn snap_scale_quarter() {
        let scale = Vec3::new(1.13, 2.38, 0.63);
        let snapped = snap_scale(scale, 0.25);
        assert!((snapped.x - 1.0).abs() < 1e-5);
        assert!((snapped.y - 2.5).abs() < 1e-5);
        assert!((snapped.z - 0.75).abs() < 1e-5);
    }

    #[test]
    fn snap_uniform_scale_test() {
        assert!((snap_uniform_scale(1.23, 0.5) - 1.0).abs() < 1e-5);
        assert!((snap_uniform_scale(1.26, 0.5) - 1.5).abs() < 1e-5);
    }

    // -- Surface snapping tests -------------------------------------------------

    #[test]
    fn snap_to_surface_flat_plane() {
        // Create a flat plane at y=0 (two triangles forming a quad).
        let triangles = vec![
            Triangle::new(
                Vec3::new(-10.0, 0.0, -10.0),
                Vec3::new(10.0, 0.0, -10.0),
                Vec3::new(10.0, 0.0, 10.0),
            ),
            Triangle::new(
                Vec3::new(-10.0, 0.0, -10.0),
                Vec3::new(10.0, 0.0, 10.0),
                Vec3::new(-10.0, 0.0, 10.0),
            ),
        ];

        let pos = Vec3::new(0.0, 5.0, 0.0);
        let hit = snap_to_surface(pos, &triangles);
        assert!(hit.is_some());

        let hit = hit.unwrap();
        assert!((hit.position.y - 0.0).abs() < 1e-4);
        assert!((hit.distance - 5.0).abs() < 1e-4);
    }

    #[test]
    fn snap_to_surface_no_hit() {
        // Triangles not below the position.
        let triangles = vec![Triangle::new(
            Vec3::new(100.0, 0.0, 100.0),
            Vec3::new(101.0, 0.0, 100.0),
            Vec3::new(100.0, 0.0, 101.0),
        )];

        let pos = Vec3::new(0.0, 5.0, 0.0);
        let hit = snap_to_surface(pos, &triangles);
        assert!(hit.is_none());
    }

    #[test]
    fn snap_to_surface_elevated_plane() {
        let triangles = vec![Triangle::new(
            Vec3::new(-10.0, 3.0, -10.0),
            Vec3::new(10.0, 3.0, -10.0),
            Vec3::new(10.0, 3.0, 10.0),
        )];

        let pos = Vec3::new(0.0, 10.0, 0.0);
        let hit = snap_to_surface(pos, &triangles).unwrap();
        assert!((hit.position.y - 3.0).abs() < 1e-4);
    }

    // -- Vertex snapping tests --------------------------------------------------

    #[test]
    fn snap_to_vertex_within_threshold() {
        let vertices = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
        ];

        let pos = Vec3::new(0.8, 0.1, 0.0);
        let snapped = snap_to_vertex(pos, &vertices, 0.5);
        assert_eq!(snapped, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn snap_to_vertex_outside_threshold() {
        let vertices = vec![Vec3::new(5.0, 0.0, 0.0)];

        let pos = Vec3::new(0.0, 0.0, 0.0);
        let snapped = snap_to_vertex(pos, &vertices, 0.5);
        // Should return original position since no vertex is close enough.
        assert_eq!(snapped, pos);
    }

    #[test]
    fn snap_to_vertex_closest() {
        let vertices = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.2, 0.0, 0.0),
        ];

        let pos = Vec3::new(1.15, 0.0, 0.0);
        let snapped = snap_to_vertex(pos, &vertices, 0.5);
        // Should snap to closer vertex (1.2).
        assert_eq!(snapped, Vec3::new(1.2, 0.0, 0.0));
    }

    // -- Edge snapping tests ----------------------------------------------------

    #[test]
    fn edge_closest_point_midpoint() {
        let edge = Edge::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let point = Vec3::new(5.0, 3.0, 0.0);
        let (closest, t) = edge.closest_point(&point);
        assert!((closest.x - 5.0).abs() < 1e-5);
        assert!((closest.y - 0.0).abs() < 1e-5);
        assert!((t - 0.5).abs() < 1e-5);
    }

    #[test]
    fn edge_closest_point_clamp_start() {
        let edge = Edge::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let point = Vec3::new(-5.0, 3.0, 0.0);
        let (closest, t) = edge.closest_point(&point);
        assert!((closest.x - 0.0).abs() < 1e-5);
        assert!((t - 0.0).abs() < 1e-5);
    }

    #[test]
    fn edge_closest_point_clamp_end() {
        let edge = Edge::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let point = Vec3::new(15.0, 3.0, 0.0);
        let (closest, t) = edge.closest_point(&point);
        assert!((closest.x - 10.0).abs() < 1e-5);
        assert!((t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn snap_to_edge_basic() {
        let edges = vec![Edge::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
        )];

        let pos = Vec3::new(5.0, 0.3, 0.0);
        let snapped = snap_to_edge(pos, &edges, 0.5);
        assert!((snapped.x - 5.0).abs() < 1e-5);
        assert!((snapped.y - 0.0).abs() < 1e-5);
    }

    #[test]
    fn snap_to_edge_too_far() {
        let edges = vec![Edge::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
        )];

        let pos = Vec3::new(5.0, 5.0, 0.0);
        let snapped = snap_to_edge(pos, &edges, 0.5);
        assert_eq!(snapped, pos); // Too far, no snap.
    }

    // -- Snap guide tests -------------------------------------------------------

    #[test]
    fn find_alignment_guides_basic() {
        let pos = Vec3::new(5.0, 3.0, 0.0);
        let refs = vec![
            Vec3::new(5.0, 10.0, 7.0), // Aligned on X.
            Vec3::new(8.0, 3.0, 2.0),  // Aligned on Y.
        ];

        let guides = find_alignment_guides(pos, &refs, 0.1, 10.0);

        // Should find X alignment with first ref and Y alignment with second.
        assert!(guides.len() >= 2);

        let x_guides: Vec<_> = guides.iter().filter(|g| g.axis == 0).collect();
        let y_guides: Vec<_> = guides.iter().filter(|g| g.axis == 1).collect();
        assert!(!x_guides.is_empty());
        assert!(!y_guides.is_empty());
    }

    #[test]
    fn find_alignment_guides_no_match() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let refs = vec![Vec3::new(10.0, 20.0, 30.0)];

        let guides = find_alignment_guides(pos, &refs, 0.01, 10.0);
        assert!(guides.is_empty());
    }

    // -- SnapSettings integration -----------------------------------------------

    #[test]
    fn apply_snap_enabled() {
        let settings = SnapSettings {
            enabled: true,
            grid_size: 0.5,
            ..Default::default()
        };
        let pos = Vec3::new(1.3, 2.7, 0.1);
        let snapped = apply_snap(pos, &settings);
        assert!((snapped.x - 1.5).abs() < 1e-5);
        assert!((snapped.y - 2.5).abs() < 1e-5);
        assert!((snapped.z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn apply_snap_disabled() {
        let settings = SnapSettings {
            enabled: false,
            ..Default::default()
        };
        let pos = Vec3::new(1.3, 2.7, 0.1);
        let snapped = apply_snap(pos, &settings);
        assert_eq!(snapped, pos);
    }

    // -- Edge helpers -----------------------------------------------------------

    #[test]
    fn edge_length() {
        let e = Edge::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 4.0, 0.0));
        assert!((e.length() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn edge_midpoint() {
        let e = Edge::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let mid = e.midpoint();
        assert!((mid.x - 5.0).abs() < 1e-5);
    }

    // -- Moller-Trumbore test ---------------------------------------------------

    #[test]
    fn ray_triangle_hit() {
        let tri = Triangle::new(
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
        );

        let origin = Vec3::new(0.0, 5.0, 0.0);
        let dir = Vec3::new(0.0, -1.0, 0.0);

        let t = ray_triangle_intersect(&origin, &dir, &tri);
        assert!(t.is_some());
        assert!((t.unwrap() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn ray_triangle_miss() {
        let tri = Triangle::new(
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(11.0, 0.0, 10.0),
            Vec3::new(10.0, 0.0, 11.0),
        );

        let origin = Vec3::new(0.0, 5.0, 0.0);
        let dir = Vec3::new(0.0, -1.0, 0.0);

        assert!(ray_triangle_intersect(&origin, &dir, &tri).is_none());
    }

    // -- SnapGuide tests --------------------------------------------------------

    #[test]
    fn snap_guide_length() {
        let guide = SnapGuide::entity_alignment(
            0,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
        );
        assert!((guide.length() - 10.0).abs() < 1e-5);
    }

    // -- Quat helpers -----------------------------------------------------------

    #[test]
    fn quat_axis_angle_roundtrip() {
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let angle = PI / 4.0;
        let q = Quat::from_axis_angle(&axis, angle);
        let (recovered_axis, recovered_angle) = q.to_axis_angle();

        assert!((recovered_angle - angle).abs() < 1e-4);
        assert!((recovered_axis.y - 1.0).abs() < 1e-4);
    }

    #[test]
    fn quat_normalize() {
        let q = Quat::new(1.0, 2.0, 3.0, 4.0);
        let n = q.normalize();
        let len = (n.x * n.x + n.y * n.y + n.z * n.z + n.w * n.w).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    // -- Vec3 helpers -----------------------------------------------------------

    #[test]
    fn vec3_operations() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);

        let sum = a.add(&b);
        assert_eq!(sum, Vec3::new(5.0, 7.0, 9.0));

        let diff = b.sub(&a);
        assert_eq!(diff, Vec3::new(3.0, 3.0, 3.0));

        let scaled = a.scale(2.0);
        assert_eq!(scaled, Vec3::new(2.0, 4.0, 6.0));

        let dot = a.dot(&b);
        assert!((dot - 32.0).abs() < 1e-5);

        let lerped = a.lerp(&b, 0.5);
        assert_eq!(lerped, Vec3::new(2.5, 3.5, 4.5));
    }
}
