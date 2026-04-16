// engine/render/src/lighting/light_culling.rs
//
// Frustum-based light culling. Tests lights against the camera frustum to
// determine which are potentially visible and worth evaluating during
// shading. Supports point lights (sphere test), spot lights (cone-sphere
// and cone-AABB tests), and directional lights (always visible).

use super::light_types::Light;
use glam::{Mat4, Vec3};

// ---------------------------------------------------------------------------
// FrustumPlane
// ---------------------------------------------------------------------------

/// A plane in 3D space, represented in Hessian normal form:
///   n.x * x + n.y * y + n.z * z + d = 0
///
/// The normal points inward (into the frustum).
#[derive(Debug, Clone, Copy)]
pub struct FrustumPlane {
    /// Inward-pointing normal.
    pub normal: Vec3,
    /// Signed distance from origin.
    pub d: f32,
}

impl FrustumPlane {
    /// Create a plane from normal and a point on the plane.
    pub fn from_normal_point(normal: Vec3, point: Vec3) -> Self {
        let n = normal.normalize_or_zero();
        Self {
            normal: n,
            d: -n.dot(point),
        }
    }

    /// Create a plane from the ABCD coefficients of the plane equation
    /// Ax + By + Cz + D = 0.
    pub fn from_abcd(a: f32, b: f32, c: f32, d: f32) -> Self {
        let len = Vec3::new(a, b, c).length();
        if len < 1e-7 {
            return Self {
                normal: Vec3::ZERO,
                d: 0.0,
            };
        }
        let inv = 1.0 / len;
        Self {
            normal: Vec3::new(a * inv, b * inv, c * inv),
            d: d * inv,
        }
    }

    /// Signed distance from a point to this plane. Positive means the point
    /// is on the inward (inside) side of the plane.
    #[inline]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }

    /// Test if a sphere is on the positive (inside) side of this plane.
    /// Returns `true` if the sphere is at least partially inside.
    #[inline]
    pub fn sphere_inside(&self, center: Vec3, radius: f32) -> bool {
        self.signed_distance(center) >= -radius
    }
}

// ---------------------------------------------------------------------------
// Frustum
// ---------------------------------------------------------------------------

/// A view frustum consisting of six planes.
///
/// Plane order: Left, Right, Bottom, Top, Near, Far.
/// All normals point inward.
#[derive(Debug, Clone)]
pub struct Frustum {
    pub planes: [FrustumPlane; 6],
}

impl Frustum {
    /// Extract a frustum from a combined view-projection matrix.
    ///
    /// Uses the Gribb-Hartmann method to extract the six frustum planes
    /// directly from the VP matrix columns.
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let r0 = vp.row(0);
        let r1 = vp.row(1);
        let r2 = vp.row(2);
        let r3 = vp.row(3);

        // Left:   row3 + row0
        let left = r3 + r0;
        // Right:  row3 - row0
        let right = r3 - r0;
        // Bottom: row3 + row1
        let bottom = r3 + r1;
        // Top:    row3 - row1
        let top = r3 - r1;
        // Near:   row3 + row2 (for RH, or row2 for LH)
        let near = r3 + r2;
        // Far:    row3 - row2
        let far = r3 - r2;

        Self {
            planes: [
                FrustumPlane::from_abcd(left.x, left.y, left.z, left.w),
                FrustumPlane::from_abcd(right.x, right.y, right.z, right.w),
                FrustumPlane::from_abcd(bottom.x, bottom.y, bottom.z, bottom.w),
                FrustumPlane::from_abcd(top.x, top.y, top.z, top.w),
                FrustumPlane::from_abcd(near.x, near.y, near.z, near.w),
                FrustumPlane::from_abcd(far.x, far.y, far.z, far.w),
            ],
        }
    }

    /// Test if a point is inside the frustum.
    pub fn contains_point(&self, point: Vec3) -> bool {
        self.planes
            .iter()
            .all(|p| p.signed_distance(point) >= 0.0)
    }

    /// Test if a sphere intersects the frustum.
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.planes.iter().all(|p| p.sphere_inside(center, radius))
    }

    /// Test if an AABB intersects the frustum.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            // Find the positive vertex (the one farthest along the plane normal).
            let p = Vec3::new(
                if plane.normal.x >= 0.0 { max.x } else { min.x },
                if plane.normal.y >= 0.0 { max.y } else { min.y },
                if plane.normal.z >= 0.0 { max.z } else { min.z },
            );
            if plane.signed_distance(p) < 0.0 {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Create an AABB from center and half-extents.
    pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Create an AABB from min and max corners.
    pub fn from_min_max(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Center of the AABB.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-extents of the AABB.
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Test if a sphere intersects this AABB.
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        let closest = Vec3::new(
            center.x.clamp(self.min.x, self.max.x),
            center.y.clamp(self.min.y, self.max.y),
            center.z.clamp(self.min.z, self.max.z),
        );
        let dist_sq = (closest - center).length_squared();
        dist_sq <= radius * radius
    }

    /// Test if a cone (defined by apex, direction, angle, and height)
    /// intersects this AABB.
    ///
    /// This is a conservative test using the cone's bounding sphere.
    pub fn intersects_cone(
        &self,
        apex: Vec3,
        direction: Vec3,
        half_angle: f32,
        height: f32,
    ) -> bool {
        // Use the bounding sphere of the cone as a conservative test.
        // The bounding sphere is centred at apex + dir * height/2 and has
        // radius = height / (2 * cos(half_angle)).
        let sphere_center = apex + direction * (height * 0.5);
        let sphere_radius = height * 0.5 / half_angle.cos().max(0.01);
        self.intersects_sphere(sphere_center, sphere_radius)
    }
}

// ---------------------------------------------------------------------------
// Light culling
// ---------------------------------------------------------------------------

/// Cull lights against a camera frustum.
///
/// Returns the indices (into the input slice) of lights that are at least
/// partially inside the frustum. Directional lights are always included.
///
/// # Arguments
/// - `frustum` — the camera view frustum.
/// - `lights` — the full set of lights to test.
///
/// # Returns
/// A vector of indices of visible lights.
pub fn cull_lights(frustum: &Frustum, lights: &[Light]) -> Vec<usize> {
    let mut visible = Vec::with_capacity(lights.len());

    for (i, light) in lights.iter().enumerate() {
        if is_light_visible(frustum, light) {
            visible.push(i);
        }
    }

    visible
}

/// Test whether a single light is visible within the frustum.
fn is_light_visible(frustum: &Frustum, light: &Light) -> bool {
    match light {
        Light::Directional(_) => {
            // Directional lights affect everything; always visible.
            true
        }
        Light::Point(p) => {
            // Test the point light's bounding sphere.
            frustum.intersects_sphere(p.position, p.radius)
        }
        Light::Spot(s) => {
            // Conservative test: use the cone's bounding sphere.
            // The bounding sphere encloses the entire cone volume.
            let half_angle = s.outer_angle;
            let sphere_center = s.position + s.direction * (s.range * 0.5);
            let sphere_radius = if half_angle >= std::f32::consts::FRAC_PI_4 {
                // Wide cone: sphere at apex with radius = range.
                s.range
            } else {
                // Narrow cone: tighter bounding sphere.
                s.range * 0.5 / half_angle.cos().max(0.01)
            };
            frustum.intersects_sphere(sphere_center, sphere_radius)
        }
        Light::Area(a) => {
            // Use the area light's bounding sphere.
            frustum.intersects_sphere(a.position, a.range)
        }
    }
}

/// Cull lights against an AABB (useful for cluster-based culling).
pub fn cull_lights_aabb(aabb: &Aabb, lights: &[Light]) -> Vec<usize> {
    let mut visible = Vec::with_capacity(lights.len());

    for (i, light) in lights.iter().enumerate() {
        let hit = match light {
            Light::Directional(_) => true,
            Light::Point(p) => aabb.intersects_sphere(p.position, p.radius),
            Light::Spot(s) => {
                aabb.intersects_cone(s.position, s.direction, s.outer_angle, s.range)
            }
            Light::Area(a) => aabb.intersects_sphere(a.position, a.range),
        };
        if hit {
            visible.push(i);
        }
    }

    visible
}

/// Cull lights against multiple AABBs in batch. Returns one list of light
/// indices per AABB.
pub fn cull_lights_batch(aabbs: &[Aabb], lights: &[Light]) -> Vec<Vec<usize>> {
    aabbs
        .iter()
        .map(|aabb| cull_lights_aabb(aabb, lights))
        .collect()
}

/// Perform a sphere-sphere intersection test.
#[inline]
pub fn sphere_sphere_intersect(
    center_a: Vec3,
    radius_a: f32,
    center_b: Vec3,
    radius_b: f32,
) -> bool {
    let dist_sq = (center_a - center_b).length_squared();
    let r_sum = radius_a + radius_b;
    dist_sq <= r_sum * r_sum
}

/// Compute the bounding sphere for a spot light cone.
///
/// Returns (center, radius) of the tightest bounding sphere that encloses
/// the cone.
pub fn spot_light_bounding_sphere(
    apex: Vec3,
    direction: Vec3,
    outer_angle: f32,
    range: f32,
) -> (Vec3, f32) {
    let cos_angle = outer_angle.cos();
    let sin_angle = outer_angle.sin();

    if cos_angle < 0.0 {
        // Very wide cone (>90 degrees): use apex as center.
        (apex, range)
    } else {
        // For narrow cones, the tightest bounding sphere is not centred
        // at the apex. Compute the optimal center and radius.
        let base_radius = range * sin_angle;

        if base_radius > range * cos_angle {
            // Wide-ish cone.
            let center = apex + direction * (base_radius * sin_angle);
            let radius = base_radius / cos_angle.max(1e-6);
            (center, radius)
        } else {
            // Narrow cone.
            let center = apex + direction * (range * cos_angle);
            (center, base_radius)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighting::light_types::{DirectionalLight, PointLight, SpotLight};

    fn identity_frustum() -> Frustum {
        let vp = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.1,
            100.0,
        );
        Frustum::from_view_projection(&vp)
    }

    #[test]
    fn frustum_contains_origin() {
        let f = identity_frustum();
        // Origin should be inside a perspective frustum looking down -Z.
        // Actually, with RH, 0,0,0 is at the camera position, which is
        // at the near plane boundary. Let's test a point slightly in front.
        assert!(f.contains_point(Vec3::new(0.0, 0.0, -1.0)));
    }

    #[test]
    fn frustum_excludes_behind() {
        let f = identity_frustum();
        assert!(!f.contains_point(Vec3::new(0.0, 0.0, 10.0)));
    }

    #[test]
    fn sphere_aabb_intersection() {
        let aabb = Aabb::from_min_max(Vec3::ZERO, Vec3::ONE);
        assert!(aabb.intersects_sphere(Vec3::new(0.5, 0.5, 0.5), 0.1));
        assert!(aabb.intersects_sphere(Vec3::new(1.5, 0.5, 0.5), 0.6));
        assert!(!aabb.intersects_sphere(Vec3::new(3.0, 3.0, 3.0), 0.5));
    }

    #[test]
    fn directional_always_visible() {
        let f = identity_frustum();
        let lights = vec![DirectionalLight::sun().to_light()];
        let visible = cull_lights(&f, &lights);
        assert_eq!(visible.len(), 1);
    }

    #[test]
    fn point_light_culling() {
        let f = identity_frustum();
        let inside = PointLight::new(Vec3::new(0.0, 0.0, -5.0), Vec3::ONE, 1.0, 2.0).to_light();
        let outside =
            PointLight::new(Vec3::new(100.0, 100.0, -5.0), Vec3::ONE, 1.0, 1.0).to_light();
        let lights = vec![inside, outside];
        let visible = cull_lights(&f, &lights);
        // Inside light should be visible, outside should not.
        assert!(visible.contains(&0));
    }

    #[test]
    fn sphere_sphere_test() {
        assert!(sphere_sphere_intersect(
            Vec3::ZERO,
            1.0,
            Vec3::new(1.5, 0.0, 0.0),
            1.0
        ));
        assert!(!sphere_sphere_intersect(
            Vec3::ZERO,
            1.0,
            Vec3::new(3.0, 0.0, 0.0),
            1.0
        ));
    }
}
