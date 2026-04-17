// engine/terrain/src/terrain_collision.rs
//
// Terrain heightfield collision detection:
//   - Ray vs heightfield using DDA (Digital Differential Analyzer)
//   - Sphere vs heightfield with cell overlap testing
//   - AABB vs heightfield
//   - Per-triangle normal computation
//   - Contact point generation
//   - Material queries per cell

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32, pub y: f32, pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    pub fn cross(self, r: Self) -> Self {
        Self::new(self.y*r.z-self.z*r.y, self.z*r.x-self.x*r.z, self.x*r.y-self.y*r.x)
    }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f32 { self.dot(self) }
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::ZERO } else { self * (1.0/l) }
    }
    pub fn min_comp(a: Self, b: Self) -> Self {
        Self::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
    }
    pub fn max_comp(a: Self, b: Self) -> Self {
        Self::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
    }
}

impl std::ops::Add for Vec3 { type Output=Self; fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z)}}
impl std::ops::Sub for Vec3 { type Output=Self; fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z)}}
impl std::ops::Mul<f32> for Vec3 { type Output=Self; fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s)}}
impl std::ops::Neg for Vec3 { type Output=Self; fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z)}}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }
}

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalized() }
    }
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

// ---------------------------------------------------------------------------
// Heightfield
// ---------------------------------------------------------------------------

/// Terrain material for a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainMaterialId(pub u16);

/// A heightfield stored as a 2D grid of height values.
#[derive(Debug, Clone)]
pub struct Heightfield {
    /// Height values in row-major order.
    pub heights: Vec<f32>,
    /// Width (number of vertices along X).
    pub width: u32,
    /// Height (number of vertices along Z).
    pub depth: u32,
    /// World-space spacing between vertices.
    pub cell_size: f32,
    /// World-space origin (corner of the heightfield).
    pub origin: Vec3,
    /// Minimum height in the heightfield.
    pub min_height: f32,
    /// Maximum height in the heightfield.
    pub max_height: f32,
    /// Per-cell material IDs (optional, (width-1)*(depth-1) entries).
    pub materials: Vec<TerrainMaterialId>,
    /// Per-cell "hole" flag (true = no collision).
    pub holes: Vec<bool>,
}

impl Heightfield {
    /// Create a flat heightfield.
    pub fn new_flat(width: u32, depth: u32, cell_size: f32, height: f32) -> Self {
        let count = (width * depth) as usize;
        let cell_count = ((width - 1) * (depth - 1)) as usize;
        Self {
            heights: vec![height; count],
            width,
            depth,
            cell_size,
            origin: Vec3::ZERO,
            min_height: height,
            max_height: height,
            materials: vec![TerrainMaterialId(0); cell_count],
            holes: vec![false; cell_count],
        }
    }

    /// Create from height data.
    pub fn from_heights(width: u32, depth: u32, cell_size: f32, heights: Vec<f32>) -> Self {
        assert_eq!(heights.len(), (width * depth) as usize);
        let min_h = heights.iter().cloned().fold(f32::MAX, f32::min);
        let max_h = heights.iter().cloned().fold(f32::MIN, f32::max);
        let cell_count = ((width - 1) * (depth - 1)) as usize;
        Self {
            heights,
            width,
            depth,
            cell_size,
            origin: Vec3::ZERO,
            min_height: min_h,
            max_height: max_h,
            materials: vec![TerrainMaterialId(0); cell_count],
            holes: vec![false; cell_count],
        }
    }

    /// Get the height at integer grid coordinates.
    #[inline]
    pub fn height_at(&self, x: u32, z: u32) -> f32 {
        if x >= self.width || z >= self.depth {
            return 0.0;
        }
        self.heights[(z * self.width + x) as usize]
    }

    /// Get the height at a world-space position using bilinear interpolation.
    pub fn sample_height(&self, world_x: f32, world_z: f32) -> f32 {
        let lx = (world_x - self.origin.x) / self.cell_size;
        let lz = (world_z - self.origin.z) / self.cell_size;

        let ix = lx.floor() as i32;
        let iz = lz.floor() as i32;
        let fx = lx - ix as f32;
        let fz = lz - iz as f32;

        let ix = ix.clamp(0, self.width as i32 - 2) as u32;
        let iz = iz.clamp(0, self.depth as i32 - 2) as u32;

        let h00 = self.height_at(ix, iz);
        let h10 = self.height_at(ix + 1, iz);
        let h01 = self.height_at(ix, iz + 1);
        let h11 = self.height_at(ix + 1, iz + 1);

        let h0 = h00 + (h10 - h00) * fx;
        let h1 = h01 + (h11 - h01) * fx;
        h0 + (h1 - h0) * fz
    }

    /// Get the world-space position of a grid vertex.
    #[inline]
    pub fn vertex_world(&self, x: u32, z: u32) -> Vec3 {
        Vec3::new(
            self.origin.x + x as f32 * self.cell_size,
            self.height_at(x, z),
            self.origin.z + z as f32 * self.cell_size,
        )
    }

    /// Get the normal at a grid cell by averaging two triangle normals.
    pub fn cell_normal(&self, cx: u32, cz: u32) -> Vec3 {
        let v00 = self.vertex_world(cx, cz);
        let v10 = self.vertex_world(cx + 1, cz);
        let v01 = self.vertex_world(cx, cz + 1);
        let v11 = self.vertex_world(cx + 1, cz + 1);

        let n1 = (v10 - v00).cross(v01 - v00).normalized();
        let n2 = (v01 - v11).cross(v10 - v11).normalized();
        ((n1 + n2) * 0.5).normalized()
    }

    /// Get the triangle normal for a specific sub-triangle within a cell.
    /// `upper` = true for the upper-left triangle, false for lower-right.
    pub fn triangle_normal(&self, cx: u32, cz: u32, upper: bool) -> Vec3 {
        let v00 = self.vertex_world(cx, cz);
        let v10 = self.vertex_world(cx + 1, cz);
        let v01 = self.vertex_world(cx, cz + 1);
        let v11 = self.vertex_world(cx + 1, cz + 1);

        if upper {
            (v10 - v00).cross(v01 - v00).normalized()
        } else {
            (v01 - v11).cross(v10 - v11).normalized()
        }
    }

    /// Check if a cell is a hole.
    pub fn is_hole(&self, cx: u32, cz: u32) -> bool {
        if cx >= self.width - 1 || cz >= self.depth - 1 {
            return true;
        }
        self.holes[(cz * (self.width - 1) + cx) as usize]
    }

    /// Get the material of a cell.
    pub fn cell_material(&self, cx: u32, cz: u32) -> TerrainMaterialId {
        if cx >= self.width - 1 || cz >= self.depth - 1 {
            return TerrainMaterialId(0);
        }
        self.materials[(cz * (self.width - 1) + cx) as usize]
    }

    /// World-space AABB of the entire heightfield.
    pub fn world_aabb(&self) -> AABB {
        AABB::new(
            Vec3::new(self.origin.x, self.min_height, self.origin.z),
            Vec3::new(
                self.origin.x + (self.width - 1) as f32 * self.cell_size,
                self.max_height,
                self.origin.z + (self.depth - 1) as f32 * self.cell_size,
            ),
        )
    }

    /// Convert world position to cell coordinates.
    #[inline]
    pub fn world_to_cell(&self, world_x: f32, world_z: f32) -> (i32, i32) {
        let lx = (world_x - self.origin.x) / self.cell_size;
        let lz = (world_z - self.origin.z) / self.cell_size;
        (lx.floor() as i32, lz.floor() as i32)
    }
}

// ---------------------------------------------------------------------------
// Contact info
// ---------------------------------------------------------------------------

/// Contact point from a terrain collision test.
#[derive(Debug, Clone, Copy)]
pub struct TerrainContact {
    pub point: Vec3,
    pub normal: Vec3,
    pub depth: f32,
    pub cell_x: u32,
    pub cell_z: u32,
    pub material: TerrainMaterialId,
}

/// Ray hit result.
#[derive(Debug, Clone, Copy)]
pub struct TerrainRayHit {
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub cell_x: u32,
    pub cell_z: u32,
    pub material: TerrainMaterialId,
}

// ---------------------------------------------------------------------------
// Ray vs heightfield (DDA)
// ---------------------------------------------------------------------------

/// Ray vs triangle intersection (Moller-Trumbore).
fn ray_triangle(ray: &Ray, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f32, Vec3)> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = ray.direction.cross(edge2);
    let a = edge1.dot(h);
    if a.abs() < 1e-8 { return None; }
    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * s.dot(h);
    if u < 0.0 || u > 1.0 { return None; }
    let q = s.cross(edge1);
    let v = f * ray.direction.dot(q);
    if v < 0.0 || u + v > 1.0 { return None; }
    let t = f * edge2.dot(q);
    if t > 1e-6 {
        let normal = edge1.cross(edge2).normalized();
        Some((t, normal))
    } else {
        None
    }
}

/// Cast a ray against the heightfield using DDA grid traversal.
pub fn raycast_heightfield(heightfield: &Heightfield, ray: &Ray, max_distance: f32) -> Option<TerrainRayHit> {
    let aabb = heightfield.world_aabb();

    // Check if ray intersects heightfield AABB.
    let (t_enter, t_exit) = ray_aabb_intersection(ray, &aabb)?;
    let t_start = t_enter.max(0.0);
    let t_end = t_exit.min(max_distance);
    if t_start >= t_end { return None; }

    let entry = ray.point_at(t_start + 0.001);
    let (mut cx, mut cz) = heightfield.world_to_cell(entry.x, entry.z);

    // DDA setup.
    let inv_cell = 1.0 / heightfield.cell_size;
    let step_x: i32 = if ray.direction.x >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if ray.direction.z >= 0.0 { 1 } else { -1 };

    let t_delta_x = if ray.direction.x.abs() > 1e-8 {
        (heightfield.cell_size / ray.direction.x.abs())
    } else {
        f32::MAX
    };
    let t_delta_z = if ray.direction.z.abs() > 1e-8 {
        (heightfield.cell_size / ray.direction.z.abs())
    } else {
        f32::MAX
    };

    let cell_origin_x = heightfield.origin.x + cx as f32 * heightfield.cell_size;
    let cell_origin_z = heightfield.origin.z + cz as f32 * heightfield.cell_size;

    let mut t_max_x = if ray.direction.x.abs() > 1e-8 {
        let boundary = if step_x > 0 {
            cell_origin_x + heightfield.cell_size
        } else {
            cell_origin_x
        };
        (boundary - ray.origin.x) / ray.direction.x
    } else {
        f32::MAX
    };

    let mut t_max_z = if ray.direction.z.abs() > 1e-8 {
        let boundary = if step_z > 0 {
            cell_origin_z + heightfield.cell_size
        } else {
            cell_origin_z
        };
        (boundary - ray.origin.z) / ray.direction.z
    } else {
        f32::MAX
    };

    let max_cells = (heightfield.width + heightfield.depth) as i32;

    for _ in 0..max_cells {
        if cx < 0 || cz < 0
            || cx >= heightfield.width as i32 - 1
            || cz >= heightfield.depth as i32 - 1
        {
            // Step to next cell.
            if t_max_x < t_max_z { cx += step_x; t_max_x += t_delta_x; }
            else { cz += step_z; t_max_z += t_delta_z; }
            continue;
        }

        let ucx = cx as u32;
        let ucz = cz as u32;

        if !heightfield.is_hole(ucx, ucz) {
            let v00 = heightfield.vertex_world(ucx, ucz);
            let v10 = heightfield.vertex_world(ucx + 1, ucz);
            let v01 = heightfield.vertex_world(ucx, ucz + 1);
            let v11 = heightfield.vertex_world(ucx + 1, ucz + 1);

            // Test both triangles in this cell.
            if let Some((t, normal)) = ray_triangle(ray, v00, v10, v01) {
                if t <= max_distance {
                    return Some(TerrainRayHit {
                        point: ray.point_at(t),
                        normal,
                        distance: t,
                        cell_x: ucx,
                        cell_z: ucz,
                        material: heightfield.cell_material(ucx, ucz),
                    });
                }
            }
            if let Some((t, normal)) = ray_triangle(ray, v10, v11, v01) {
                if t <= max_distance {
                    return Some(TerrainRayHit {
                        point: ray.point_at(t),
                        normal,
                        distance: t,
                        cell_x: ucx,
                        cell_z: ucz,
                        material: heightfield.cell_material(ucx, ucz),
                    });
                }
            }
        }

        // Advance DDA.
        if t_max_x < t_max_z {
            if t_max_x > max_distance { break; }
            cx += step_x;
            t_max_x += t_delta_x;
        } else {
            if t_max_z > max_distance { break; }
            cz += step_z;
            t_max_z += t_delta_z;
        }
    }

    None
}

fn ray_aabb_intersection(ray: &Ray, aabb: &AABB) -> Option<(f32, f32)> {
    let inv_dx = if ray.direction.x.abs() > 1e-10 { 1.0/ray.direction.x } else { f32::MAX };
    let inv_dy = if ray.direction.y.abs() > 1e-10 { 1.0/ray.direction.y } else { f32::MAX };
    let inv_dz = if ray.direction.z.abs() > 1e-10 { 1.0/ray.direction.z } else { f32::MAX };

    let t1 = (aabb.min.x - ray.origin.x) * inv_dx;
    let t2 = (aabb.max.x - ray.origin.x) * inv_dx;
    let t3 = (aabb.min.y - ray.origin.y) * inv_dy;
    let t4 = (aabb.max.y - ray.origin.y) * inv_dy;
    let t5 = (aabb.min.z - ray.origin.z) * inv_dz;
    let t6 = (aabb.max.z - ray.origin.z) * inv_dz;

    let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
    let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

    if tmax < 0.0 || tmin > tmax { None } else { Some((tmin, tmax)) }
}

// ---------------------------------------------------------------------------
// Sphere vs heightfield
// ---------------------------------------------------------------------------

/// Test a sphere against the heightfield, returning contact points.
pub fn sphere_heightfield(
    heightfield: &Heightfield,
    center: Vec3,
    radius: f32,
) -> Vec<TerrainContact> {
    let mut contacts = Vec::new();

    // Determine which cells the sphere might overlap.
    let min_x = ((center.x - radius - heightfield.origin.x) / heightfield.cell_size).floor() as i32;
    let max_x = ((center.x + radius - heightfield.origin.x) / heightfield.cell_size).ceil() as i32;
    let min_z = ((center.z - radius - heightfield.origin.z) / heightfield.cell_size).floor() as i32;
    let max_z = ((center.z + radius - heightfield.origin.z) / heightfield.cell_size).ceil() as i32;

    let min_x = min_x.max(0) as u32;
    let max_x = (max_x as u32).min(heightfield.width - 2);
    let min_z = min_z.max(0) as u32;
    let max_z = (max_z as u32).min(heightfield.depth - 2);

    for cz in min_z..=max_z {
        for cx in min_x..=max_x {
            if heightfield.is_hole(cx, cz) { continue; }

            let v00 = heightfield.vertex_world(cx, cz);
            let v10 = heightfield.vertex_world(cx + 1, cz);
            let v01 = heightfield.vertex_world(cx, cz + 1);
            let v11 = heightfield.vertex_world(cx + 1, cz + 1);

            // Test sphere against both triangles.
            if let Some(contact) = sphere_triangle(center, radius, v00, v10, v01) {
                contacts.push(TerrainContact {
                    point: contact.0,
                    normal: contact.1,
                    depth: contact.2,
                    cell_x: cx,
                    cell_z: cz,
                    material: heightfield.cell_material(cx, cz),
                });
            }
            if let Some(contact) = sphere_triangle(center, radius, v10, v11, v01) {
                contacts.push(TerrainContact {
                    point: contact.0,
                    normal: contact.1,
                    depth: contact.2,
                    cell_x: cx,
                    cell_z: cz,
                    material: heightfield.cell_material(cx, cz),
                });
            }
        }
    }

    contacts
}

/// Sphere vs triangle collision. Returns (contact_point, normal, penetration_depth).
fn sphere_triangle(center: Vec3, radius: f32, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(Vec3, Vec3, f32)> {
    // Find closest point on triangle to sphere center.
    let closest = closest_point_on_triangle(center, v0, v1, v2);
    let diff = center - closest;
    let dist_sq = diff.length_sq();

    if dist_sq > radius * radius {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-8 {
        diff * (1.0 / dist)
    } else {
        (v1 - v0).cross(v2 - v0).normalized()
    };

    let depth = radius - dist;
    Some((closest, normal, depth))
}

/// Closest point on triangle to a point.
fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 { return a; }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 { return b; }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 { return c; }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

// ---------------------------------------------------------------------------
// AABB vs heightfield
// ---------------------------------------------------------------------------

/// Test an AABB against the heightfield.
pub fn aabb_heightfield(heightfield: &Heightfield, aabb: &AABB) -> Vec<TerrainContact> {
    let mut contacts = Vec::new();

    let min_x = ((aabb.min.x - heightfield.origin.x) / heightfield.cell_size).floor() as i32;
    let max_x = ((aabb.max.x - heightfield.origin.x) / heightfield.cell_size).ceil() as i32;
    let min_z = ((aabb.min.z - heightfield.origin.z) / heightfield.cell_size).floor() as i32;
    let max_z = ((aabb.max.z - heightfield.origin.z) / heightfield.cell_size).ceil() as i32;

    let min_x = min_x.max(0) as u32;
    let max_x = (max_x as u32).min(heightfield.width - 2);
    let min_z = min_z.max(0) as u32;
    let max_z = (max_z as u32).min(heightfield.depth - 2);

    let center = Vec3::new(
        (aabb.min.x + aabb.max.x) * 0.5,
        (aabb.min.y + aabb.max.y) * 0.5,
        (aabb.min.z + aabb.max.z) * 0.5,
    );

    for cz in min_z..=max_z {
        for cx in min_x..=max_x {
            if heightfield.is_hole(cx, cz) { continue; }

            // Sample height at AABB center projected onto this cell.
            let wx = heightfield.origin.x + (cx as f32 + 0.5) * heightfield.cell_size;
            let wz = heightfield.origin.z + (cz as f32 + 0.5) * heightfield.cell_size;
            let h = heightfield.sample_height(
                wx.clamp(aabb.min.x, aabb.max.x),
                wz.clamp(aabb.min.z, aabb.max.z),
            );

            if aabb.min.y < h {
                let depth = h - aabb.min.y;
                let normal = heightfield.cell_normal(cx, cz);
                contacts.push(TerrainContact {
                    point: Vec3::new(wx, h, wz),
                    normal,
                    depth,
                    cell_x: cx,
                    cell_z: cz,
                    material: heightfield.cell_material(cx, cz),
                });
            }
        }
    }

    contacts
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_heightfield() -> Heightfield {
        Heightfield::new_flat(10, 10, 1.0, 0.0)
    }

    fn make_sloped_heightfield() -> Heightfield {
        let mut hf = Heightfield::new_flat(10, 10, 1.0, 0.0);
        for z in 0..10u32 {
            for x in 0..10u32 {
                hf.heights[(z * 10 + x) as usize] = x as f32 * 0.5;
            }
        }
        hf.max_height = 4.5;
        hf
    }

    #[test]
    fn test_sample_height() {
        let hf = make_flat_heightfield();
        assert!((hf.sample_height(5.0, 5.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_raycast_down() {
        let hf = make_flat_heightfield();
        let ray = Ray::new(Vec3::new(4.5, 10.0, 4.5), Vec3::new(0.0, -1.0, 0.0));
        let hit = raycast_heightfield(&hf, &ray, 100.0);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.point.y - 0.0).abs() < 0.1);
        assert!((h.distance - 10.0).abs() < 0.2);
    }

    #[test]
    fn test_raycast_miss() {
        let hf = make_flat_heightfield();
        // Ray going up, should miss.
        let ray = Ray::new(Vec3::new(4.5, 10.0, 4.5), Vec3::new(0.0, 1.0, 0.0));
        let hit = raycast_heightfield(&hf, &ray, 100.0);
        assert!(hit.is_none());
    }

    #[test]
    fn test_sphere_collision() {
        let hf = make_flat_heightfield();
        let contacts = sphere_heightfield(&hf, Vec3::new(4.5, 0.3, 4.5), 0.5);
        assert!(!contacts.is_empty());
        assert!(contacts[0].depth > 0.0);
    }

    #[test]
    fn test_sphere_no_collision() {
        let hf = make_flat_heightfield();
        let contacts = sphere_heightfield(&hf, Vec3::new(4.5, 5.0, 4.5), 0.5);
        assert!(contacts.is_empty());
    }

    #[test]
    fn test_aabb_collision() {
        let hf = make_flat_heightfield();
        let aabb = AABB::new(Vec3::new(3.0, -0.5, 3.0), Vec3::new(5.0, 0.5, 5.0));
        let contacts = aabb_heightfield(&hf, &aabb);
        assert!(!contacts.is_empty());
    }

    #[test]
    fn test_cell_normal() {
        let hf = make_flat_heightfield();
        let n = hf.cell_normal(4, 4);
        assert!((n.y - 1.0).abs() < 0.01); // flat terrain, normal should be up
    }

    #[test]
    fn test_holes() {
        let mut hf = make_flat_heightfield();
        hf.holes[4 * 9 + 4] = true;
        let contacts = sphere_heightfield(&hf, Vec3::new(4.5, 0.3, 4.5), 0.5);
        // Should not detect collision at hole
        let hole_contacts: Vec<_> = contacts.iter()
            .filter(|c| c.cell_x == 4 && c.cell_z == 4)
            .collect();
        assert!(hole_contacts.is_empty());
    }
}
