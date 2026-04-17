// engine/render/src/mesh_generation.rs
//
// Procedural mesh generation: parametric surfaces (sphere, torus, spring,
// Mobius strip), revolution surfaces, extrusion, Catmull-Clark subdivision,
// edge loop insertion, normal/tangent computation, and UV generation.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 { pub x: f32, pub y: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self { x: self.y*r.z - self.z*r.y, y: self.z*r.x - self.x*r.z, z: self.x*r.y - self.y*r.x } }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn normalize(self) -> Self { let l = self.length(); if l < 1e-12 { Self::ZERO } else { Self { x:self.x/l, y:self.y/l, z:self.z/l } } }
    pub fn scale(self, s: f32) -> Self { Self { x:self.x*s, y:self.y*s, z:self.z*s } }
    pub fn add(self, r: Self) -> Self { Self { x:self.x+r.x, y:self.y+r.y, z:self.z+r.z } }
    pub fn sub(self, r: Self) -> Self { Self { x:self.x-r.x, y:self.y-r.y, z:self.z-r.z } }
    pub fn neg(self) -> Self { Self { x:-self.x, y:-self.y, z:-self.z } }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
}

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }
}

// ---------------------------------------------------------------------------
// Mesh data structure
// ---------------------------------------------------------------------------

/// A vertex with position, normal, tangent, and UV.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub tangent: Vec4,  // xyz = tangent direction, w = handedness (+1 or -1)
    pub uv: Vec2,
}

impl Vertex {
    pub fn new(pos: Vec3, normal: Vec3, uv: Vec2) -> Self {
        Self { position: pos, normal, tangent: Vec4::new(1.0, 0.0, 0.0, 1.0), uv }
    }
}

/// Generated mesh data.
#[derive(Debug, Clone)]
pub struct GeneratedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub name: String,
}

impl GeneratedMesh {
    pub fn new(name: &str) -> Self {
        Self { vertices: Vec::new(), indices: Vec::new(), name: name.to_string() }
    }

    /// Compute normals from triangle faces (area-weighted smooth normals).
    pub fn compute_normals(&mut self) {
        // Zero out normals
        for v in &mut self.vertices {
            v.normal = Vec3::ZERO;
        }

        // Accumulate face normals
        let tri_count = self.indices.len() / 3;
        for i in 0..tri_count {
            let i0 = self.indices[i * 3] as usize;
            let i1 = self.indices[i * 3 + 1] as usize;
            let i2 = self.indices[i * 3 + 2] as usize;

            let v0 = self.vertices[i0].position;
            let v1 = self.vertices[i1].position;
            let v2 = self.vertices[i2].position;

            let edge1 = v1.sub(v0);
            let edge2 = v2.sub(v0);
            let face_normal = edge1.cross(edge2); // not normalized = area weighted

            self.vertices[i0].normal = self.vertices[i0].normal.add(face_normal);
            self.vertices[i1].normal = self.vertices[i1].normal.add(face_normal);
            self.vertices[i2].normal = self.vertices[i2].normal.add(face_normal);
        }

        // Normalize
        for v in &mut self.vertices {
            v.normal = v.normal.normalize();
        }
    }

    /// Compute tangents using the MikkTSpace algorithm (simplified).
    pub fn compute_tangents(&mut self) {
        let vert_count = self.vertices.len();
        let mut tangent_accum = vec![Vec3::ZERO; vert_count];
        let mut bitangent_accum = vec![Vec3::ZERO; vert_count];

        let tri_count = self.indices.len() / 3;
        for i in 0..tri_count {
            let i0 = self.indices[i * 3] as usize;
            let i1 = self.indices[i * 3 + 1] as usize;
            let i2 = self.indices[i * 3 + 2] as usize;

            let v0 = &self.vertices[i0];
            let v1 = &self.vertices[i1];
            let v2 = &self.vertices[i2];

            let dp1 = v1.position.sub(v0.position);
            let dp2 = v2.position.sub(v0.position);
            let duv1 = Vec2::new(v1.uv.x - v0.uv.x, v1.uv.y - v0.uv.y);
            let duv2 = Vec2::new(v2.uv.x - v0.uv.x, v2.uv.y - v0.uv.y);

            let r = 1.0 / (duv1.x * duv2.y - duv1.y * duv2.x).max(1e-7);

            let tangent = Vec3::new(
                (duv2.y * dp1.x - duv1.y * dp2.x) * r,
                (duv2.y * dp1.y - duv1.y * dp2.y) * r,
                (duv2.y * dp1.z - duv1.y * dp2.z) * r,
            );

            let bitangent = Vec3::new(
                (duv1.x * dp2.x - duv2.x * dp1.x) * r,
                (duv1.x * dp2.y - duv2.x * dp1.y) * r,
                (duv1.x * dp2.z - duv2.x * dp1.z) * r,
            );

            tangent_accum[i0] = tangent_accum[i0].add(tangent);
            tangent_accum[i1] = tangent_accum[i1].add(tangent);
            tangent_accum[i2] = tangent_accum[i2].add(tangent);
            bitangent_accum[i0] = bitangent_accum[i0].add(bitangent);
            bitangent_accum[i1] = bitangent_accum[i1].add(bitangent);
            bitangent_accum[i2] = bitangent_accum[i2].add(bitangent);
        }

        for i in 0..vert_count {
            let n = self.vertices[i].normal;
            let t = tangent_accum[i];
            let b = bitangent_accum[i];

            // Gram-Schmidt orthogonalize
            let ortho_t = t.sub(n.scale(n.dot(t))).normalize();

            // Handedness
            let w = if n.cross(t).dot(b) < 0.0 { -1.0 } else { 1.0 };

            self.vertices[i].tangent = Vec4::new(ortho_t.x, ortho_t.y, ortho_t.z, w);
        }
    }

    /// Merge with another mesh.
    pub fn merge(&mut self, other: &GeneratedMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        for &idx in &other.indices {
            self.indices.push(base + idx);
        }
    }

    /// Triangle count.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Flip all normals and winding order.
    pub fn flip_normals(&mut self) {
        for v in &mut self.vertices {
            v.normal = v.normal.neg();
        }
        // Reverse winding
        let tri_count = self.indices.len() / 3;
        for i in 0..tri_count {
            self.indices.swap(i * 3 + 1, i * 3 + 2);
        }
    }

    /// Apply a uniform scale.
    pub fn scale(&mut self, s: f32) {
        for v in &mut self.vertices {
            v.position = v.position.scale(s);
        }
    }

    /// Translate all vertices.
    pub fn translate(&mut self, offset: Vec3) {
        for v in &mut self.vertices {
            v.position = v.position.add(offset);
        }
    }
}

// ---------------------------------------------------------------------------
// Parametric surfaces
// ---------------------------------------------------------------------------

/// Generate a UV sphere.
pub fn generate_sphere(radius: f32, segments: u32, rings: u32) -> GeneratedMesh {
    let mut mesh = GeneratedMesh::new("sphere");

    for ring in 0..=rings {
        let theta = ring as f32 / rings as f32 * PI;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for seg in 0..=segments {
            let phi = seg as f32 / segments as f32 * 2.0 * PI;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let x = sin_theta * cos_phi;
            let y = cos_theta;
            let z = sin_theta * sin_phi;

            let normal = Vec3::new(x, y, z);
            let position = normal.scale(radius);
            let uv = Vec2::new(seg as f32 / segments as f32, ring as f32 / rings as f32);

            mesh.vertices.push(Vertex::new(position, normal, uv));
        }
    }

    for ring in 0..rings {
        for seg in 0..segments {
            let a = ring * (segments + 1) + seg;
            let b = a + segments + 1;

            mesh.indices.push(a);
            mesh.indices.push(b);
            mesh.indices.push(a + 1);

            mesh.indices.push(b);
            mesh.indices.push(b + 1);
            mesh.indices.push(a + 1);
        }
    }

    mesh.compute_tangents();
    mesh
}

/// Generate a torus.
pub fn generate_torus(major_radius: f32, minor_radius: f32, major_segments: u32, minor_segments: u32) -> GeneratedMesh {
    let mut mesh = GeneratedMesh::new("torus");

    for i in 0..=major_segments {
        let u = i as f32 / major_segments as f32 * 2.0 * PI;
        let cos_u = u.cos();
        let sin_u = u.sin();

        for j in 0..=minor_segments {
            let v = j as f32 / minor_segments as f32 * 2.0 * PI;
            let cos_v = v.cos();
            let sin_v = v.sin();

            let x = (major_radius + minor_radius * cos_v) * cos_u;
            let y = minor_radius * sin_v;
            let z = (major_radius + minor_radius * cos_v) * sin_u;

            let nx = cos_v * cos_u;
            let ny = sin_v;
            let nz = cos_v * sin_u;

            let position = Vec3::new(x, y, z);
            let normal = Vec3::new(nx, ny, nz).normalize();
            let uv = Vec2::new(i as f32 / major_segments as f32, j as f32 / minor_segments as f32);

            mesh.vertices.push(Vertex::new(position, normal, uv));
        }
    }

    for i in 0..major_segments {
        for j in 0..minor_segments {
            let a = i * (minor_segments + 1) + j;
            let b = a + minor_segments + 1;

            mesh.indices.push(a);
            mesh.indices.push(b);
            mesh.indices.push(a + 1);

            mesh.indices.push(b);
            mesh.indices.push(b + 1);
            mesh.indices.push(a + 1);
        }
    }

    mesh.compute_tangents();
    mesh
}

/// Generate a spring / helix surface.
pub fn generate_spring(
    coil_radius: f32,
    wire_radius: f32,
    num_coils: f32,
    segments_per_coil: u32,
    wire_segments: u32,
    pitch: f32,
) -> GeneratedMesh {
    let mut mesh = GeneratedMesh::new("spring");

    let total_segments = (num_coils * segments_per_coil as f32) as u32;

    for i in 0..=total_segments {
        let t = i as f32 / total_segments as f32;
        let angle = t * num_coils * 2.0 * PI;

        // Center of the wire cross-section
        let cx = coil_radius * angle.cos();
        let cy = t * num_coils * pitch;
        let cz = coil_radius * angle.sin();

        // Tangent along the coil
        let tx = -coil_radius * angle.sin();
        let ty = pitch * num_coils / total_segments as f32;
        let tz = coil_radius * angle.cos();
        let tangent = Vec3::new(tx, ty, tz).normalize();

        // Normal and binormal
        let up = Vec3::UP;
        let binormal = tangent.cross(up).normalize();
        let normal = binormal.cross(tangent).normalize();

        for j in 0..=wire_segments {
            let v_angle = j as f32 / wire_segments as f32 * 2.0 * PI;

            let px = cx + wire_radius * (v_angle.cos() * normal.x + v_angle.sin() * binormal.x);
            let py = cy + wire_radius * (v_angle.cos() * normal.y + v_angle.sin() * binormal.y);
            let pz = cz + wire_radius * (v_angle.cos() * normal.z + v_angle.sin() * binormal.z);

            let n = Vec3::new(
                v_angle.cos() * normal.x + v_angle.sin() * binormal.x,
                v_angle.cos() * normal.y + v_angle.sin() * binormal.y,
                v_angle.cos() * normal.z + v_angle.sin() * binormal.z,
            ).normalize();

            mesh.vertices.push(Vertex::new(
                Vec3::new(px, py, pz),
                n,
                Vec2::new(t, j as f32 / wire_segments as f32),
            ));
        }
    }

    for i in 0..total_segments {
        for j in 0..wire_segments {
            let a = i * (wire_segments + 1) + j;
            let b = a + wire_segments + 1;

            mesh.indices.push(a);
            mesh.indices.push(b);
            mesh.indices.push(a + 1);

            mesh.indices.push(b);
            mesh.indices.push(b + 1);
            mesh.indices.push(a + 1);
        }
    }

    mesh.compute_tangents();
    mesh
}

/// Generate a Mobius strip.
pub fn generate_mobius(radius: f32, width: f32, segments: u32, width_segments: u32) -> GeneratedMesh {
    let mut mesh = GeneratedMesh::new("mobius");

    for i in 0..=segments {
        let u = i as f32 / segments as f32;
        let theta = u * 2.0 * PI;

        for j in 0..=width_segments {
            let v = (j as f32 / width_segments as f32 - 0.5) * width;

            // Half-twist: the surface normal rotates by PI as we go around
            let half_theta = theta * 0.5;

            let x = (radius + v * half_theta.cos()) * theta.cos();
            let y = v * half_theta.sin();
            let z = (radius + v * half_theta.cos()) * theta.sin();

            // Approximate normal via cross product of partial derivatives
            let du = 0.01;
            let theta2 = (u + du) * 2.0 * PI;
            let ht2 = theta2 * 0.5;
            let x2 = (radius + v * ht2.cos()) * theta2.cos();
            let y2 = v * ht2.sin();
            let z2 = (radius + v * ht2.cos()) * theta2.sin();

            let dv = 0.01;
            let v2 = v + dv;
            let x3 = (radius + v2 * half_theta.cos()) * theta.cos();
            let y3 = v2 * half_theta.sin();
            let z3 = (radius + v2 * half_theta.cos()) * theta.sin();

            let du_vec = Vec3::new(x2 - x, y2 - y, z2 - z);
            let dv_vec = Vec3::new(x3 - x, y3 - y, z3 - z);
            let normal = du_vec.cross(dv_vec).normalize();

            mesh.vertices.push(Vertex::new(
                Vec3::new(x, y, z),
                normal,
                Vec2::new(u, j as f32 / width_segments as f32),
            ));
        }
    }

    for i in 0..segments {
        for j in 0..width_segments {
            let a = i * (width_segments + 1) + j;
            let b = a + width_segments + 1;

            mesh.indices.push(a);
            mesh.indices.push(b);
            mesh.indices.push(a + 1);

            mesh.indices.push(b);
            mesh.indices.push(b + 1);
            mesh.indices.push(a + 1);
        }
    }

    mesh.compute_tangents();
    mesh
}

/// Generate a cylinder.
pub fn generate_cylinder(radius: f32, height: f32, segments: u32, height_segments: u32, caps: bool) -> GeneratedMesh {
    let mut mesh = GeneratedMesh::new("cylinder");

    // Side
    for ring in 0..=height_segments {
        let y = (ring as f32 / height_segments as f32 - 0.5) * height;
        for seg in 0..=segments {
            let angle = seg as f32 / segments as f32 * 2.0 * PI;
            let x = angle.cos() * radius;
            let z = angle.sin() * radius;

            mesh.vertices.push(Vertex::new(
                Vec3::new(x, y, z),
                Vec3::new(angle.cos(), 0.0, angle.sin()),
                Vec2::new(seg as f32 / segments as f32, ring as f32 / height_segments as f32),
            ));
        }
    }

    for ring in 0..height_segments {
        for seg in 0..segments {
            let a = ring * (segments + 1) + seg;
            let b = a + segments + 1;
            mesh.indices.push(a);
            mesh.indices.push(b);
            mesh.indices.push(a + 1);
            mesh.indices.push(b);
            mesh.indices.push(b + 1);
            mesh.indices.push(a + 1);
        }
    }

    // Caps
    if caps {
        for side in 0..2 {
            let y = if side == 0 { -height * 0.5 } else { height * 0.5 };
            let ny = if side == 0 { -1.0 } else { 1.0 };
            let base = mesh.vertices.len() as u32;

            // Center vertex
            mesh.vertices.push(Vertex::new(
                Vec3::new(0.0, y, 0.0),
                Vec3::new(0.0, ny, 0.0),
                Vec2::new(0.5, 0.5),
            ));

            for seg in 0..=segments {
                let angle = seg as f32 / segments as f32 * 2.0 * PI;
                mesh.vertices.push(Vertex::new(
                    Vec3::new(angle.cos() * radius, y, angle.sin() * radius),
                    Vec3::new(0.0, ny, 0.0),
                    Vec2::new(angle.cos() * 0.5 + 0.5, angle.sin() * 0.5 + 0.5),
                ));
            }

            for seg in 0..segments {
                if side == 0 {
                    mesh.indices.push(base);
                    mesh.indices.push(base + seg + 2);
                    mesh.indices.push(base + seg + 1);
                } else {
                    mesh.indices.push(base);
                    mesh.indices.push(base + seg + 1);
                    mesh.indices.push(base + seg + 2);
                }
            }
        }
    }

    mesh.compute_tangents();
    mesh
}

/// Generate a plane.
pub fn generate_plane(width: f32, depth: f32, width_segments: u32, depth_segments: u32) -> GeneratedMesh {
    let mut mesh = GeneratedMesh::new("plane");

    for z in 0..=depth_segments {
        for x in 0..=width_segments {
            let u = x as f32 / width_segments as f32;
            let v = z as f32 / depth_segments as f32;
            mesh.vertices.push(Vertex::new(
                Vec3::new((u - 0.5) * width, 0.0, (v - 0.5) * depth),
                Vec3::UP,
                Vec2::new(u, v),
            ));
        }
    }

    for z in 0..depth_segments {
        for x in 0..width_segments {
            let a = z * (width_segments + 1) + x;
            let b = a + width_segments + 1;
            mesh.indices.push(a);
            mesh.indices.push(b);
            mesh.indices.push(a + 1);
            mesh.indices.push(b);
            mesh.indices.push(b + 1);
            mesh.indices.push(a + 1);
        }
    }

    mesh.compute_tangents();
    mesh
}

/// Generate a box / cube.
pub fn generate_box(width: f32, height: f32, depth: f32) -> GeneratedMesh {
    let mut mesh = GeneratedMesh::new("box");
    let hw = width * 0.5;
    let hh = height * 0.5;
    let hd = depth * 0.5;

    // 6 faces, each as a quad with unique normals
    let faces = [
        // +X
        ([hw, -hh, -hd], [hw, hh, -hd], [hw, hh, hd], [hw, -hh, hd], [1.0, 0.0, 0.0]),
        // -X
        ([-hw, -hh, hd], [-hw, hh, hd], [-hw, hh, -hd], [-hw, -hh, -hd], [-1.0, 0.0, 0.0]),
        // +Y
        ([-hw, hh, -hd], [-hw, hh, hd], [hw, hh, hd], [hw, hh, -hd], [0.0, 1.0, 0.0]),
        // -Y
        ([-hw, -hh, hd], [-hw, -hh, -hd], [hw, -hh, -hd], [hw, -hh, hd], [0.0, -1.0, 0.0]),
        // +Z
        ([-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd], [0.0, 0.0, 1.0]),
        // -Z
        ([hw, -hh, -hd], [-hw, -hh, -hd], [-hw, hh, -hd], [hw, hh, -hd], [0.0, 0.0, -1.0]),
    ];

    let uvs = [Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0)];

    for (p0, p1, p2, p3, n) in &faces {
        let base = mesh.vertices.len() as u32;
        let normal = Vec3::new(n[0], n[1], n[2]);

        mesh.vertices.push(Vertex::new(Vec3::new(p0[0], p0[1], p0[2]), normal, uvs[0]));
        mesh.vertices.push(Vertex::new(Vec3::new(p1[0], p1[1], p1[2]), normal, uvs[1]));
        mesh.vertices.push(Vertex::new(Vec3::new(p2[0], p2[1], p2[2]), normal, uvs[2]));
        mesh.vertices.push(Vertex::new(Vec3::new(p3[0], p3[1], p3[2]), normal, uvs[3]));

        mesh.indices.push(base);
        mesh.indices.push(base + 1);
        mesh.indices.push(base + 2);
        mesh.indices.push(base);
        mesh.indices.push(base + 2);
        mesh.indices.push(base + 3);
    }

    mesh.compute_tangents();
    mesh
}

// ---------------------------------------------------------------------------
// Revolution and extrusion
// ---------------------------------------------------------------------------

/// Generate a surface of revolution by rotating a profile curve around the Y axis.
pub fn generate_revolution(profile: &[Vec2], segments: u32) -> GeneratedMesh {
    let mut mesh = GeneratedMesh::new("revolution");
    let profile_len = profile.len();

    for seg in 0..=segments {
        let angle = seg as f32 / segments as f32 * 2.0 * PI;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for (j, pt) in profile.iter().enumerate() {
            let x = pt.x * cos_a;
            let z = pt.x * sin_a;
            let y = pt.y;

            // Compute normal from profile tangent
            let tangent_2d = if j == 0 {
                Vec2::new(profile[1].x - profile[0].x, profile[1].y - profile[0].y)
            } else if j == profile_len - 1 {
                Vec2::new(profile[j].x - profile[j-1].x, profile[j].y - profile[j-1].y)
            } else {
                Vec2::new(profile[j+1].x - profile[j-1].x, profile[j+1].y - profile[j-1].y)
            };

            // Normal is perpendicular to tangent in the profile plane
            let profile_normal = Vec2::new(tangent_2d.y, -tangent_2d.x);
            let len = (profile_normal.x * profile_normal.x + profile_normal.y * profile_normal.y).sqrt();
            let pn = if len > 1e-6 { Vec2::new(profile_normal.x / len, profile_normal.y / len) } else { Vec2::new(1.0, 0.0) };

            let normal = Vec3::new(pn.x * cos_a, pn.y, pn.x * sin_a).normalize();

            mesh.vertices.push(Vertex::new(
                Vec3::new(x, y, z),
                normal,
                Vec2::new(seg as f32 / segments as f32, j as f32 / (profile_len - 1) as f32),
            ));
        }
    }

    for seg in 0..segments {
        for j in 0..(profile_len - 1) {
            let a = (seg * profile_len as u32 + j as u32) as u32;
            let b = a + profile_len as u32;

            mesh.indices.push(a);
            mesh.indices.push(b);
            mesh.indices.push(a + 1);
            mesh.indices.push(b);
            mesh.indices.push(b + 1);
            mesh.indices.push(a + 1);
        }
    }

    mesh.compute_tangents();
    mesh
}

/// Extrude a 2D polygon along a path.
pub fn generate_extrusion(polygon: &[Vec2], path: &[Vec3], closed: bool) -> GeneratedMesh {
    let mut mesh = GeneratedMesh::new("extrusion");
    let poly_len = polygon.len();
    let path_len = path.len();

    for (pi, &path_pt) in path.iter().enumerate() {
        // Compute frame (tangent, normal, binormal) at this path point
        let tangent = if pi == 0 {
            path[1].sub(path[0]).normalize()
        } else if pi == path_len - 1 {
            path[pi].sub(path[pi - 1]).normalize()
        } else {
            path[pi + 1].sub(path[pi - 1]).normalize()
        };

        let up = if tangent.dot(Vec3::UP).abs() > 0.99 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::UP
        };

        let binormal = tangent.cross(up).normalize();
        let normal = binormal.cross(tangent).normalize();

        let t = pi as f32 / (path_len - 1).max(1) as f32;

        for (vi, pt) in polygon.iter().enumerate() {
            let world_pos = path_pt
                .add(binormal.scale(pt.x))
                .add(normal.scale(pt.y));

            let vert_normal = binormal.scale(pt.x).add(normal.scale(pt.y)).normalize();

            mesh.vertices.push(Vertex::new(
                world_pos,
                vert_normal,
                Vec2::new(vi as f32 / poly_len as f32, t),
            ));
        }
    }

    let segments = if closed { path_len } else { path_len - 1 };
    for pi in 0..segments {
        let next = if pi + 1 >= path_len { 0 } else { pi + 1 };
        for vi in 0..poly_len {
            let next_v = (vi + 1) % poly_len;
            let a = (pi * poly_len + vi) as u32;
            let b = (next * poly_len + vi) as u32;
            let c = (pi * poly_len + next_v) as u32;
            let d = (next * poly_len + next_v) as u32;

            mesh.indices.push(a);
            mesh.indices.push(b);
            mesh.indices.push(c);
            mesh.indices.push(b);
            mesh.indices.push(d);
            mesh.indices.push(c);
        }
    }

    mesh.compute_normals();
    mesh.compute_tangents();
    mesh
}

// ---------------------------------------------------------------------------
// Catmull-Clark subdivision
// ---------------------------------------------------------------------------

/// Apply one level of Catmull-Clark subdivision.
pub fn subdivide_catmull_clark(mesh: &GeneratedMesh) -> GeneratedMesh {
    let mut result = GeneratedMesh::new(&format!("{}_subdivided", mesh.name));

    // Build adjacency
    let vert_count = mesh.vertices.len();
    let tri_count = mesh.indices.len() / 3;

    // Step 1: Compute face points (centroid of each face)
    let mut face_points = Vec::with_capacity(tri_count);
    for f in 0..tri_count {
        let i0 = mesh.indices[f * 3] as usize;
        let i1 = mesh.indices[f * 3 + 1] as usize;
        let i2 = mesh.indices[f * 3 + 2] as usize;

        let fp = mesh.vertices[i0].position
            .add(mesh.vertices[i1].position)
            .add(mesh.vertices[i2].position)
            .scale(1.0 / 3.0);
        face_points.push(fp);
    }

    // Step 2: Find edges and compute edge points
    let mut edge_map: std::collections::HashMap<(u32, u32), Vec<usize>> = std::collections::HashMap::new();

    for f in 0..tri_count {
        let indices = [mesh.indices[f*3], mesh.indices[f*3+1], mesh.indices[f*3+2]];
        for e in 0..3 {
            let v0 = indices[e];
            let v1 = indices[(e + 1) % 3];
            let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            edge_map.entry(key).or_default().push(f);
        }
    }

    let mut edge_points: std::collections::HashMap<(u32, u32), Vec3> = std::collections::HashMap::new();

    for (&(v0, v1), faces) in &edge_map {
        let mid = mesh.vertices[v0 as usize].position
            .add(mesh.vertices[v1 as usize].position)
            .scale(0.5);

        if faces.len() == 2 {
            // Interior edge: average of edge midpoint and adjacent face points
            let fp_avg = face_points[faces[0]]
                .add(face_points[faces[1]])
                .scale(0.5);
            edge_points.insert((v0, v1), mid.lerp(fp_avg, 0.5));
        } else {
            // Boundary edge: just midpoint
            edge_points.insert((v0, v1), mid);
        }
    }

    // Step 3: Compute new vertex positions
    let mut vert_faces: Vec<Vec<usize>> = vec![Vec::new(); vert_count];
    let mut vert_edges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); vert_count];

    for f in 0..tri_count {
        for e in 0..3 {
            let vi = mesh.indices[f * 3 + e] as usize;
            vert_faces[vi].push(f);
        }
    }

    for &(v0, v1) in edge_map.keys() {
        vert_edges[v0 as usize].push((v0, v1));
        vert_edges[v1 as usize].push((v0, v1));
    }

    let mut new_vertex_positions = Vec::with_capacity(vert_count);
    for vi in 0..vert_count {
        let n = vert_faces[vi].len() as f32;
        if n < 1.0 {
            new_vertex_positions.push(mesh.vertices[vi].position);
            continue;
        }

        // Average of adjacent face points
        let mut f_avg = Vec3::ZERO;
        for &fi in &vert_faces[vi] {
            f_avg = f_avg.add(face_points[fi]);
        }
        f_avg = f_avg.scale(1.0 / n);

        // Average of midpoints of adjacent edges
        let mut e_avg = Vec3::ZERO;
        let edge_count = vert_edges[vi].len() as f32;
        for &(ev0, ev1) in &vert_edges[vi] {
            let mid = mesh.vertices[ev0 as usize].position
                .add(mesh.vertices[ev1 as usize].position)
                .scale(0.5);
            e_avg = e_avg.add(mid);
        }
        if edge_count > 0.0 {
            e_avg = e_avg.scale(1.0 / edge_count);
        }

        // New position
        let old_pos = mesh.vertices[vi].position;
        let new_pos = f_avg.add(e_avg.scale(2.0)).add(old_pos.scale(n - 3.0)).scale(1.0 / n);
        new_vertex_positions.push(new_pos);
    }

    // Step 4: Build the subdivided mesh
    // For each original face, create sub-faces connecting:
    //   - Original vertex (new position)
    //   - Edge points on adjacent edges
    //   - Face point

    // Add all new vertex positions
    for vi in 0..vert_count {
        let uv = mesh.vertices[vi].uv;
        result.vertices.push(Vertex::new(new_vertex_positions[vi], Vec3::UP, uv));
    }

    // Add face points
    let face_point_base = result.vertices.len() as u32;
    for (fi, &fp) in face_points.iter().enumerate() {
        let f_indices = [mesh.indices[fi*3] as usize, mesh.indices[fi*3+1] as usize, mesh.indices[fi*3+2] as usize];
        let uv = Vec2::new(
            (mesh.vertices[f_indices[0]].uv.x + mesh.vertices[f_indices[1]].uv.x + mesh.vertices[f_indices[2]].uv.x) / 3.0,
            (mesh.vertices[f_indices[0]].uv.y + mesh.vertices[f_indices[1]].uv.y + mesh.vertices[f_indices[2]].uv.y) / 3.0,
        );
        result.vertices.push(Vertex::new(fp, Vec3::UP, uv));
    }

    // Add edge points
    let mut edge_point_indices: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for (&key, ep) in &edge_points {
        let idx = result.vertices.len() as u32;
        let uv = Vec2::new(
            (mesh.vertices[key.0 as usize].uv.x + mesh.vertices[key.1 as usize].uv.x) * 0.5,
            (mesh.vertices[key.0 as usize].uv.y + mesh.vertices[key.1 as usize].uv.y) * 0.5,
        );
        result.vertices.push(Vertex::new(*ep, Vec3::UP, uv));
        edge_point_indices.insert(key, idx);
    }

    // Create sub-faces
    for fi in 0..tri_count {
        let v = [mesh.indices[fi*3], mesh.indices[fi*3+1], mesh.indices[fi*3+2]];
        let fp_idx = face_point_base + fi as u32;

        for e in 0..3 {
            let v0 = v[e];
            let v1 = v[(e + 1) % 3];
            let edge_key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            let v2 = v[(e + 2) % 3];
            let edge_key2 = if v2 < v0 { (v2, v0) } else { (v0, v2) };

            if let (Some(&ep1), Some(&ep2)) = (edge_point_indices.get(&edge_key), edge_point_indices.get(&edge_key2)) {
                // Triangle: v0 -> ep1 -> fp -> ep2
                result.indices.push(v0);
                result.indices.push(ep1);
                result.indices.push(fp_idx);

                result.indices.push(v0);
                result.indices.push(fp_idx);
                result.indices.push(ep2);
            }
        }
    }

    result.compute_normals();
    result.compute_tangents();
    result
}

// ---------------------------------------------------------------------------
// Edge loop insertion
// ---------------------------------------------------------------------------

/// Insert an edge loop along a ring of edges at a parametric position.
pub fn insert_edge_loop(mesh: &mut GeneratedMesh, edge_ring: &[(u32, u32)], t: f32) {
    let mut new_vert_map: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();

    for &(v0, v1) in edge_ring {
        let pos = mesh.vertices[v0 as usize].position
            .lerp(mesh.vertices[v1 as usize].position, t);
        let normal = mesh.vertices[v0 as usize].normal
            .lerp(mesh.vertices[v1 as usize].normal, t)
            .normalize();
        let uv = Vec2::new(
            mesh.vertices[v0 as usize].uv.x + (mesh.vertices[v1 as usize].uv.x - mesh.vertices[v0 as usize].uv.x) * t,
            mesh.vertices[v0 as usize].uv.y + (mesh.vertices[v1 as usize].uv.y - mesh.vertices[v0 as usize].uv.y) * t,
        );

        let new_idx = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex::new(pos, normal, uv));
        let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        new_vert_map.insert(key, new_idx);
    }

    // Split triangles that contain these edges
    let mut new_indices = Vec::new();
    let tri_count = mesh.indices.len() / 3;

    for f in 0..tri_count {
        let i0 = mesh.indices[f * 3];
        let i1 = mesh.indices[f * 3 + 1];
        let i2 = mesh.indices[f * 3 + 2];

        let edges = [(i0, i1), (i1, i2), (i2, i0)];
        let mut split_edge = None;
        let mut split_vert = 0u32;

        for (ei, &(ea, eb)) in edges.iter().enumerate() {
            let key = if ea < eb { (ea, eb) } else { (eb, ea) };
            if let Some(&new_v) = new_vert_map.get(&key) {
                split_edge = Some(ei);
                split_vert = new_v;
                break;
            }
        }

        if let Some(ei) = split_edge {
            let tri = [i0, i1, i2];
            let a = tri[ei];
            let b = tri[(ei + 1) % 3];
            let c = tri[(ei + 2) % 3];

            // Split into 2 triangles
            new_indices.push(a);
            new_indices.push(split_vert);
            new_indices.push(c);

            new_indices.push(split_vert);
            new_indices.push(b);
            new_indices.push(c);
        } else {
            new_indices.push(i0);
            new_indices.push(i1);
            new_indices.push(i2);
        }
    }

    mesh.indices = new_indices;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_generation() {
        let mesh = generate_sphere(1.0, 16, 8);
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn test_torus_generation() {
        let mesh = generate_torus(2.0, 0.5, 16, 8);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_box_generation() {
        let mesh = generate_box(1.0, 1.0, 1.0);
        assert_eq!(mesh.vertices.len(), 24); // 6 faces * 4 verts
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn test_plane_generation() {
        let mesh = generate_plane(10.0, 10.0, 4, 4);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_revolution() {
        let profile = vec![Vec2::new(1.0, -1.0), Vec2::new(1.5, 0.0), Vec2::new(1.0, 1.0)];
        let mesh = generate_revolution(&profile, 16);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_subdivision() {
        let mesh = generate_box(1.0, 1.0, 1.0);
        let subdivided = subdivide_catmull_clark(&mesh);
        assert!(subdivided.vertices.len() > mesh.vertices.len());
    }

    #[test]
    fn test_compute_normals() {
        let mut mesh = generate_plane(1.0, 1.0, 1, 1);
        mesh.compute_normals();
        for v in &mesh.vertices {
            assert!((v.normal.length() - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_mobius() {
        let mesh = generate_mobius(2.0, 0.5, 32, 4);
        assert!(mesh.triangle_count() > 0);
    }
}
