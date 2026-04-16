// engine/render/src/trails.rs
//
// Trail / ribbon renderer for the Genovo engine. Generates smooth ribbon
// geometry behind moving objects using Catmull-Rom spline interpolation.
//
// A trail is defined by a series of *trail points* sampled over time. Each
// point records position, width, color, and timestamp. Old points fade and
// are eventually removed. The renderer builds a triangle strip mesh from
// the live points, with UV coordinates running along the trail length.

use glam::Vec3;

// ---------------------------------------------------------------------------
// TrailPoint
// ---------------------------------------------------------------------------

/// A single sample point along a trail ribbon.
#[derive(Debug, Clone, Copy)]
pub struct TrailPoint {
    /// World-space position.
    pub position: Vec3,
    /// Width of the trail at this point.
    pub width: f32,
    /// RGBA color at this point.
    pub color: [f32; 4],
    /// Time when this point was created (seconds since trail start).
    pub time: f32,
}

// ---------------------------------------------------------------------------
// TrailVertex
// ---------------------------------------------------------------------------

/// A vertex in the generated trail mesh.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TrailVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// Texture coordinates (u = across width, v = along length).
    pub uv: [f32; 2],
    /// RGBA color.
    pub color: [f32; 4],
}

// ---------------------------------------------------------------------------
// TrailTextureMode
// ---------------------------------------------------------------------------

/// How UVs are generated along the trail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrailTextureMode {
    /// UV V coordinate is based on trail length in world units.
    WorldLength,
    /// UV V coordinate is normalized to [0, 1] across the entire trail.
    Normalized,
    /// UV V coordinate tiles at a fixed repeat rate.
    Tile { repeats_per_unit: u32 },
}

impl Default for TrailTextureMode {
    fn default() -> Self {
        TrailTextureMode::Normalized
    }
}

// ---------------------------------------------------------------------------
// TrailRenderer
// ---------------------------------------------------------------------------

/// Generates a ribbon trail mesh behind a moving object.
///
/// # Usage
///
/// ```ignore
/// let mut trail = TrailRenderer::new(2.0, 0.5);
/// // Each frame:
/// trail.add_point(object_position, current_time);
/// trail.update(current_time);
/// let vertices = trail.vertices();
/// let indices = trail.indices();
/// ```
pub struct TrailRenderer {
    /// Raw trail points (before interpolation).
    points: Vec<TrailPoint>,
    /// Maximum trail lifetime in seconds. Points older than this are removed.
    pub lifetime: f32,
    /// Width of the trail at the head (newest point).
    pub start_width: f32,
    /// Width of the trail at the tail (oldest point).
    pub end_width: f32,
    /// Start color (head).
    pub start_color: [f32; 4],
    /// End color (tail).
    pub end_color: [f32; 4],
    /// Minimum distance between sampled points. Closer points are skipped
    /// to avoid overlapping geometry.
    pub min_vertex_distance: f32,
    /// Number of interpolation steps between each pair of control points.
    pub interpolation_steps: u32,
    /// Texture mode.
    pub texture_mode: TrailTextureMode,
    /// If `true`, use Catmull-Rom spline interpolation. Otherwise, linear.
    pub smooth: bool,
    /// Generated vertex buffer (reused each frame).
    vertex_buffer: Vec<TrailVertex>,
    /// Generated index buffer (reused each frame).
    index_buffer: Vec<u32>,
    /// Camera up direction (for billboard orientation of the ribbon).
    camera_up: Vec3,
    /// Whether the trail is currently emitting (adding new points).
    pub emitting: bool,
}

impl TrailRenderer {
    /// Creates a new trail renderer.
    ///
    /// # Arguments
    /// * `lifetime` - How long trail points persist (seconds).
    /// * `width` - Initial trail width.
    pub fn new(lifetime: f32, width: f32) -> Self {
        Self {
            points: Vec::with_capacity(256),
            lifetime,
            start_width: width,
            end_width: 0.0,
            start_color: [1.0, 1.0, 1.0, 1.0],
            end_color: [1.0, 1.0, 1.0, 0.0],
            min_vertex_distance: 0.05,
            interpolation_steps: 4,
            texture_mode: TrailTextureMode::Normalized,
            smooth: true,
            vertex_buffer: Vec::with_capacity(512),
            index_buffer: Vec::with_capacity(768),
            camera_up: Vec3::Y,
            emitting: true,
        }
    }

    /// Sets the width taper (start to end).
    pub fn with_width(mut self, start: f32, end: f32) -> Self {
        self.start_width = start;
        self.end_width = end;
        self
    }

    /// Sets the color gradient (start to end).
    pub fn with_color(mut self, start: [f32; 4], end: [f32; 4]) -> Self {
        self.start_color = start;
        self.end_color = end;
        self
    }

    /// Sets the minimum distance between sample points.
    pub fn with_min_distance(mut self, dist: f32) -> Self {
        self.min_vertex_distance = dist;
        self
    }

    /// Sets the interpolation quality.
    pub fn with_interpolation(mut self, steps: u32) -> Self {
        self.interpolation_steps = steps;
        self
    }

    /// Sets the texture mode.
    pub fn with_texture_mode(mut self, mode: TrailTextureMode) -> Self {
        self.texture_mode = mode;
        self
    }

    /// Enables or disables smooth (Catmull-Rom) interpolation.
    pub fn with_smooth(mut self, smooth: bool) -> Self {
        self.smooth = smooth;
        self
    }

    /// Adds a new trail point at the current position.
    ///
    /// Points that are too close to the previous point (less than
    /// `min_vertex_distance`) are skipped.
    pub fn add_point(&mut self, position: Vec3, time: f32) {
        if !self.emitting {
            return;
        }

        // Check minimum distance.
        if let Some(last) = self.points.last() {
            let dist = (position - last.position).length();
            if dist < self.min_vertex_distance {
                // Update the last point's position instead of adding a new one.
                let last_idx = self.points.len() - 1;
                self.points[last_idx].position = position;
                self.points[last_idx].time = time;
                return;
            }
        }

        self.points.push(TrailPoint {
            position,
            width: self.start_width,
            color: self.start_color,
            time,
        });
    }

    /// Updates the trail: removes expired points and regenerates the mesh.
    ///
    /// # Arguments
    /// * `current_time` - The current simulation time.
    /// * `camera_pos` - Camera position (for billboard orientation).
    pub fn update(&mut self, current_time: f32, camera_pos: Vec3) {
        // Remove expired points.
        let cutoff = current_time - self.lifetime;
        self.points.retain(|p| p.time >= cutoff);

        if self.points.len() < 2 {
            self.vertex_buffer.clear();
            self.index_buffer.clear();
            return;
        }

        // Update width and color based on age.
        let oldest_time = self.points[0].time;
        let newest_time = self.points[self.points.len() - 1].time;
        let time_range = (newest_time - oldest_time).max(1e-6);

        for point in &mut self.points {
            let t = (point.time - oldest_time) / time_range;
            // t=0 is the oldest (tail), t=1 is the newest (head).
            point.width = lerp(self.end_width, self.start_width, t);
            point.color = lerp_color(&self.end_color, &self.start_color, t);

            // Additional alpha fade based on remaining lifetime.
            let age = current_time - point.time;
            let life_frac = (age / self.lifetime).clamp(0.0, 1.0);
            point.color[3] *= 1.0 - life_frac;
        }

        // Generate mesh.
        if self.smooth && self.points.len() >= 4 {
            self.generate_smooth_mesh(camera_pos);
        } else {
            self.generate_linear_mesh(camera_pos);
        }
    }

    /// Generates the trail mesh with linear interpolation.
    fn generate_linear_mesh(&mut self, camera_pos: Vec3) {
        self.vertex_buffer.clear();
        self.index_buffer.clear();

        if self.points.len() < 2 {
            return;
        }

        let point_count = self.points.len();
        self.vertex_buffer.reserve(point_count * 2);
        self.index_buffer.reserve((point_count - 1) * 6);

        let total_length = self.compute_total_length();

        let mut accumulated_length = 0.0;

        for i in 0..point_count {
            let point = &self.points[i];

            // Compute tangent direction.
            let tangent = if i == 0 {
                (self.points[1].position - self.points[0].position).normalize()
            } else if i == point_count - 1 {
                (self.points[i].position - self.points[i - 1].position).normalize()
            } else {
                (self.points[i + 1].position - self.points[i - 1].position).normalize()
            };

            // Compute right vector (perpendicular to tangent and view direction).
            let view_dir = (camera_pos - point.position).normalize();
            let right = tangent.cross(view_dir);
            let right_len = right.length();
            let right = if right_len > 1e-6 {
                right / right_len
            } else {
                // Degenerate case.
                let alt = if tangent.y.abs() < 0.99 {
                    Vec3::Y
                } else {
                    Vec3::X
                };
                tangent.cross(alt).normalize()
            };

            let half_width = point.width * 0.5;

            // Two vertices: left and right.
            let left = point.position - right * half_width;
            let right_pos = point.position + right * half_width;

            // Compute V coordinate.
            if i > 0 {
                accumulated_length +=
                    (self.points[i].position - self.points[i - 1].position).length();
            }
            let v = match self.texture_mode {
                TrailTextureMode::Normalized => {
                    if total_length > 0.0 {
                        accumulated_length / total_length
                    } else {
                        0.0
                    }
                }
                TrailTextureMode::WorldLength => accumulated_length,
                TrailTextureMode::Tile { repeats_per_unit } => {
                    accumulated_length * repeats_per_unit as f32
                }
            };

            self.vertex_buffer.push(TrailVertex {
                position: [left.x, left.y, left.z],
                uv: [0.0, v],
                color: point.color,
            });
            self.vertex_buffer.push(TrailVertex {
                position: [right_pos.x, right_pos.y, right_pos.z],
                uv: [1.0, v],
                color: point.color,
            });
        }

        // Generate triangle strip indices.
        for i in 0..(point_count - 1) as u32 {
            let base = i * 2;
            // Triangle 1.
            self.index_buffer.push(base);
            self.index_buffer.push(base + 1);
            self.index_buffer.push(base + 2);
            // Triangle 2.
            self.index_buffer.push(base + 1);
            self.index_buffer.push(base + 3);
            self.index_buffer.push(base + 2);
        }
    }

    /// Generates the trail mesh with Catmull-Rom spline interpolation.
    fn generate_smooth_mesh(&mut self, camera_pos: Vec3) {
        self.vertex_buffer.clear();
        self.index_buffer.clear();

        let n = self.points.len();
        if n < 2 {
            return;
        }

        // Generate interpolated points.
        let mut interp_points: Vec<TrailPoint> = Vec::new();
        let steps = self.interpolation_steps.max(1);

        for i in 0..n - 1 {
            // Get four control points for Catmull-Rom.
            let p0 = if i == 0 { 0 } else { i - 1 };
            let p1 = i;
            let p2 = i + 1;
            let p3 = (i + 2).min(n - 1);

            let cp0 = &self.points[p0];
            let cp1 = &self.points[p1];
            let cp2 = &self.points[p2];
            let cp3 = &self.points[p3];

            for s in 0..steps {
                let t = s as f32 / steps as f32;

                let position = catmull_rom(
                    cp0.position,
                    cp1.position,
                    cp2.position,
                    cp3.position,
                    t,
                );

                let width = catmull_rom_scalar(cp0.width, cp1.width, cp2.width, cp3.width, t);
                let time = lerp(cp1.time, cp2.time, t);
                let color = lerp_color(&cp1.color, &cp2.color, t);

                interp_points.push(TrailPoint {
                    position,
                    width,
                    color,
                    time,
                });
            }
        }
        // Add the last point.
        interp_points.push(self.points[n - 1]);

        // Now generate the mesh from interpolated points.
        let interp_count = interp_points.len();
        if interp_count < 2 {
            return;
        }

        self.vertex_buffer.reserve(interp_count * 2);
        self.index_buffer.reserve((interp_count - 1) * 6);

        // Compute total length for UV mapping.
        let total_length: f32 = (1..interp_count)
            .map(|i| (interp_points[i].position - interp_points[i - 1].position).length())
            .sum();

        let mut accumulated_length = 0.0;

        for i in 0..interp_count {
            let point = &interp_points[i];

            let tangent = if i == 0 {
                (interp_points[1].position - interp_points[0].position).normalize()
            } else if i == interp_count - 1 {
                (interp_points[i].position - interp_points[i - 1].position).normalize()
            } else {
                (interp_points[i + 1].position - interp_points[i - 1].position).normalize()
            };

            let view_dir = (camera_pos - point.position).normalize();
            let right = tangent.cross(view_dir);
            let right_len = right.length();
            let right = if right_len > 1e-6 {
                right / right_len
            } else {
                let alt = if tangent.y.abs() < 0.99 {
                    Vec3::Y
                } else {
                    Vec3::X
                };
                tangent.cross(alt).normalize()
            };

            let half_width = point.width * 0.5;
            let left = point.position - right * half_width;
            let right_pos = point.position + right * half_width;

            if i > 0 {
                accumulated_length +=
                    (interp_points[i].position - interp_points[i - 1].position).length();
            }
            let v = match self.texture_mode {
                TrailTextureMode::Normalized => {
                    if total_length > 0.0 {
                        accumulated_length / total_length
                    } else {
                        0.0
                    }
                }
                TrailTextureMode::WorldLength => accumulated_length,
                TrailTextureMode::Tile { repeats_per_unit } => {
                    accumulated_length * repeats_per_unit as f32
                }
            };

            self.vertex_buffer.push(TrailVertex {
                position: [left.x, left.y, left.z],
                uv: [0.0, v],
                color: point.color,
            });
            self.vertex_buffer.push(TrailVertex {
                position: [right_pos.x, right_pos.y, right_pos.z],
                uv: [1.0, v],
                color: point.color,
            });
        }

        for i in 0..(interp_count - 1) as u32 {
            let base = i * 2;
            self.index_buffer.push(base);
            self.index_buffer.push(base + 1);
            self.index_buffer.push(base + 2);
            self.index_buffer.push(base + 1);
            self.index_buffer.push(base + 3);
            self.index_buffer.push(base + 2);
        }
    }

    /// Computes the total arc length of the trail.
    fn compute_total_length(&self) -> f32 {
        let mut length = 0.0;
        for i in 1..self.points.len() {
            length += (self.points[i].position - self.points[i - 1].position).length();
        }
        length
    }

    /// Returns the generated vertices.
    pub fn vertices(&self) -> &[TrailVertex] {
        &self.vertex_buffer
    }

    /// Returns the generated indices.
    pub fn indices(&self) -> &[u32] {
        &self.index_buffer
    }

    /// Returns the number of trail points.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Returns the number of generated vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertex_buffer.len()
    }

    /// Clears all trail points and generated geometry.
    pub fn clear(&mut self) {
        self.points.clear();
        self.vertex_buffer.clear();
        self.index_buffer.clear();
    }

    /// Returns the bounding box of the trail.
    pub fn compute_aabb(&self) -> (Vec3, Vec3) {
        if self.points.is_empty() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let mut min = self.points[0].position;
        let mut max = self.points[0].position;
        for point in &self.points[1..] {
            min = min.min(point.position);
            max = max.max(point.position);
        }
        let max_width = self.start_width.max(self.end_width);
        let expand = Vec3::splat(max_width * 0.5);
        (min - expand, max + expand)
    }
}

// ---------------------------------------------------------------------------
// Catmull-Rom spline interpolation
// ---------------------------------------------------------------------------

/// Catmull-Rom spline interpolation for Vec3.
///
/// Given four control points (p0, p1, p2, p3), interpolates between p1 and p2
/// at parameter `t` in [0, 1].
pub fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom matrix coefficients (tension = 0.5).
    let a = p1 * 2.0;
    let b = p2 - p0;
    let c = p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3;
    let d = -p0 + p1 * 3.0 - p2 * 3.0 + p3;

    (a + b * t + c * t2 + d * t3) * 0.5
}

/// Catmull-Rom interpolation for a scalar value.
pub fn catmull_rom_scalar(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;

    let a = p1 * 2.0;
    let b = p2 - p0;
    let c = p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3;
    let d = -p0 + p1 * 3.0 - p2 * 3.0 + p3;

    (a + b * t + c * t2 + d * t3) * 0.5
}

/// Evaluates the tangent (first derivative) of a Catmull-Rom spline.
pub fn catmull_rom_tangent(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;

    let b = p2 - p0;
    let c = p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3;
    let d = -p0 + p1 * 3.0 - p2 * 3.0 + p3;

    (b + c * (2.0 * t) + d * (3.0 * t2)) * 0.5
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp_color(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trail_basic_generation() {
        let mut trail = TrailRenderer::new(5.0, 1.0);
        trail.smooth = false;

        // Add some points.
        for i in 0..10 {
            let t = i as f32 * 0.1;
            trail.add_point(Vec3::new(i as f32, 0.0, 0.0), t);
        }

        trail.update(0.9, Vec3::new(5.0, 5.0, 0.0));

        assert!(trail.vertex_count() > 0);
        assert!(trail.indices().len() > 0);
    }

    #[test]
    fn trail_smooth_generation() {
        let mut trail = TrailRenderer::new(5.0, 1.0)
            .with_interpolation(8);

        for i in 0..10 {
            let t = i as f32 * 0.1;
            let angle = t * std::f32::consts::PI;
            trail.add_point(
                Vec3::new(angle.cos() * 5.0, 0.0, angle.sin() * 5.0),
                t,
            );
        }

        trail.update(0.9, Vec3::new(0.0, 5.0, 0.0));

        // Smooth trail should have more vertices than raw points.
        assert!(
            trail.vertex_count() > 20,
            "Smooth trail should generate more vertices, got {}",
            trail.vertex_count()
        );
    }

    #[test]
    fn trail_expiry() {
        let mut trail = TrailRenderer::new(0.5, 1.0);
        trail.smooth = false;

        for i in 0..10 {
            trail.add_point(Vec3::new(i as f32, 0.0, 0.0), i as f32 * 0.1);
        }

        // Update at time 2.0 -- all points older than 1.5 should be gone.
        trail.update(2.0, Vec3::ZERO);
        assert!(
            trail.point_count() < 10,
            "Old points should be removed, got {}",
            trail.point_count()
        );
    }

    #[test]
    fn catmull_rom_endpoints() {
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(1.0, 0.0, 0.0);
        let p2 = Vec3::new(2.0, 1.0, 0.0);
        let p3 = Vec3::new(3.0, 1.0, 0.0);

        let start = catmull_rom(p0, p1, p2, p3, 0.0);
        let end = catmull_rom(p0, p1, p2, p3, 1.0);

        assert!((start - p1).length() < 0.01, "t=0 should be at p1");
        assert!((end - p2).length() < 0.01, "t=1 should be at p2");
    }

    #[test]
    fn min_vertex_distance() {
        let mut trail = TrailRenderer::new(5.0, 1.0)
            .with_min_distance(1.0);

        // Add two points very close together.
        trail.add_point(Vec3::new(0.0, 0.0, 0.0), 0.0);
        trail.add_point(Vec3::new(0.1, 0.0, 0.0), 0.1);
        assert_eq!(trail.point_count(), 1, "Close point should update, not add");

        trail.add_point(Vec3::new(2.0, 0.0, 0.0), 0.2);
        assert_eq!(trail.point_count(), 2, "Far point should be added");
    }
}
