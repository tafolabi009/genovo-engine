// engine/render/src/mesh.rs
//
// Mesh system for the Genovo engine. Provides mesh data structures, flexible
// vertex formats, procedural primitive generation, LOD management, and a
// builder API for programmatic mesh construction.

use glam::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Vertex format
// ---------------------------------------------------------------------------

/// Supported vertex attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexAttribute {
    Position,    // vec3<f32>
    Normal,      // vec3<f32>
    Tangent,     // vec4<f32> (xyz = tangent, w = handedness)
    UV0,         // vec2<f32>
    UV1,         // vec2<f32>
    Color,       // vec4<f32>
    BoneWeights, // vec4<f32>
    BoneIndices, // vec4<u32>
}

impl VertexAttribute {
    /// Size of this attribute in bytes.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Position => 12,    // 3 * f32
            Self::Normal => 12,
            Self::Tangent => 16,     // 4 * f32
            Self::UV0 => 8,          // 2 * f32
            Self::UV1 => 8,
            Self::Color => 16,       // 4 * f32
            Self::BoneWeights => 16,
            Self::BoneIndices => 16, // 4 * u32
        }
    }

    /// Number of float components.
    pub fn component_count(&self) -> usize {
        match self {
            Self::Position | Self::Normal => 3,
            Self::UV0 | Self::UV1 => 2,
            Self::Tangent | Self::Color | Self::BoneWeights | Self::BoneIndices => 4,
        }
    }
}

/// Describes the vertex layout for a mesh.
#[derive(Debug, Clone)]
pub struct VertexFormat {
    /// Ordered list of vertex attributes.
    pub attributes: Vec<VertexAttribute>,
    /// Stride in bytes (total size of one vertex).
    pub stride: usize,
    /// Byte offset of each attribute within the vertex.
    pub offsets: Vec<usize>,
}

impl VertexFormat {
    /// Create a vertex format from a list of attributes.
    pub fn new(attributes: &[VertexAttribute]) -> Self {
        let mut offsets = Vec::with_capacity(attributes.len());
        let mut offset = 0usize;
        for attr in attributes {
            offsets.push(offset);
            offset += attr.byte_size();
        }
        Self {
            attributes: attributes.to_vec(),
            stride: offset,
            offsets,
        }
    }

    /// Standard PBR vertex format: position, normal, tangent, UV0.
    pub fn standard() -> Self {
        Self::new(&[
            VertexAttribute::Position,
            VertexAttribute::Normal,
            VertexAttribute::Tangent,
            VertexAttribute::UV0,
        ])
    }

    /// Full vertex format with all attributes.
    pub fn full() -> Self {
        Self::new(&[
            VertexAttribute::Position,
            VertexAttribute::Normal,
            VertexAttribute::Tangent,
            VertexAttribute::UV0,
            VertexAttribute::UV1,
            VertexAttribute::Color,
        ])
    }

    /// Skinned vertex format.
    pub fn skinned() -> Self {
        Self::new(&[
            VertexAttribute::Position,
            VertexAttribute::Normal,
            VertexAttribute::Tangent,
            VertexAttribute::UV0,
            VertexAttribute::BoneWeights,
            VertexAttribute::BoneIndices,
        ])
    }

    /// Position-only format (for shadow passes).
    pub fn position_only() -> Self {
        Self::new(&[VertexAttribute::Position])
    }

    /// Whether the format has a given attribute.
    pub fn has_attribute(&self, attr: VertexAttribute) -> bool {
        self.attributes.contains(&attr)
    }

    /// Get the byte offset of an attribute, if present.
    pub fn attribute_offset(&self, attr: VertexAttribute) -> Option<usize> {
        self.attributes.iter().position(|a| *a == attr).map(|i| self.offsets[i])
    }
}

// ---------------------------------------------------------------------------
// Standard vertex
// ---------------------------------------------------------------------------

/// A standard PBR vertex with position, normal, tangent, and UV.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3, tangent: Vec4, uv: Vec2) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            tangent: tangent.to_array(),
            uv: uv.to_array(),
        }
    }

    /// Byte size of a single vertex.
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for AABB {
    fn default() -> Self {
        Self { min: Vec3::splat(f32::MAX), max: Vec3::splat(f32::MIN) }
    }
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    /// Expand the AABB to include a point.
    pub fn expand_point(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Expand to include another AABB.
    pub fn expand_aabb(&mut self, other: &AABB) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Center of the AABB.
    pub fn center(&self) -> Vec3 { (self.min + self.max) * 0.5 }

    /// Half-extents.
    pub fn half_extents(&self) -> Vec3 { (self.max - self.min) * 0.5 }

    /// Full extents (size).
    pub fn size(&self) -> Vec3 { self.max - self.min }

    /// Radius of the bounding sphere.
    pub fn radius(&self) -> f32 { self.half_extents().length() }

    /// Surface area.
    pub fn surface_area(&self) -> f32 {
        let s = self.size();
        2.0 * (s.x * s.y + s.y * s.z + s.z * s.x)
    }

    /// Test if a point is inside.
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x && point.x <= self.max.x
            && point.y >= self.min.y && point.y <= self.max.y
            && point.z >= self.min.z && point.z <= self.max.z
    }

    /// Test intersection with another AABB.
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }
}

// ---------------------------------------------------------------------------
// SubMesh
// ---------------------------------------------------------------------------

/// A contiguous range of indices within a mesh, associated with a material.
#[derive(Debug, Clone)]
pub struct SubMesh {
    /// Byte offset into the index buffer (or first index).
    pub index_offset: u32,
    /// Number of indices.
    pub index_count: u32,
    /// Material index for this submesh.
    pub material_index: u32,
    /// Local AABB for this submesh.
    pub bounds: AABB,
}

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

/// A renderable mesh with vertex data, index data, and bounds.
pub struct Mesh {
    /// Vertex data (packed according to the vertex format).
    pub vertices: Vec<Vertex>,
    /// Index data (u32).
    pub indices: Vec<u32>,
    /// Total vertex count.
    pub vertex_count: u32,
    /// Total index count.
    pub index_count: u32,
    /// Bounding box of the entire mesh.
    pub bounds: AABB,
    /// Submeshes (material groups).
    pub submeshes: Vec<SubMesh>,
    /// Vertex format description.
    pub format: VertexFormat,
}

impl Mesh {
    /// Create a mesh from vertex and index data.
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        let vertex_count = vertices.len() as u32;
        let index_count = indices.len() as u32;

        let mut bounds = AABB::default();
        for v in &vertices {
            bounds.expand_point(Vec3::from_array(v.position));
        }

        Self {
            vertices, indices, vertex_count, index_count, bounds,
            submeshes: vec![SubMesh {
                index_offset: 0, index_count, material_index: 0,
                bounds,
            }],
            format: VertexFormat::standard(),
        }
    }

    /// Get vertex data as bytes.
    pub fn vertex_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.vertices)
    }

    /// Get index data as bytes.
    pub fn index_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.indices)
    }

    /// Recompute the bounding box from vertex positions.
    pub fn recompute_bounds(&mut self) {
        self.bounds = AABB::default();
        for v in &self.vertices {
            self.bounds.expand_point(Vec3::from_array(v.position));
        }
    }

    /// Total number of triangles.
    pub fn triangle_count(&self) -> u32 { self.index_count / 3 }

    /// Compute flat normals from triangle faces.
    pub fn compute_flat_normals(&mut self) {
        for i in (0..self.indices.len()).step_by(3) {
            if i + 2 >= self.indices.len() { break; }
            let i0 = self.indices[i] as usize;
            let i1 = self.indices[i + 1] as usize;
            let i2 = self.indices[i + 2] as usize;

            let p0 = Vec3::from_array(self.vertices[i0].position);
            let p1 = Vec3::from_array(self.vertices[i1].position);
            let p2 = Vec3::from_array(self.vertices[i2].position);

            let normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            let n = normal.to_array();

            self.vertices[i0].normal = n;
            self.vertices[i1].normal = n;
            self.vertices[i2].normal = n;
        }
    }

    /// Compute tangents from UV data using MikkTSpace-compatible algorithm.
    pub fn compute_tangents(&mut self) {
        // Reset tangents.
        for v in &mut self.vertices {
            v.tangent = [0.0, 0.0, 0.0, 1.0];
        }

        let mut tan1 = vec![Vec3::ZERO; self.vertices.len()];
        let mut tan2 = vec![Vec3::ZERO; self.vertices.len()];

        for i in (0..self.indices.len()).step_by(3) {
            if i + 2 >= self.indices.len() { break; }
            let i0 = self.indices[i] as usize;
            let i1 = self.indices[i + 1] as usize;
            let i2 = self.indices[i + 2] as usize;

            let p0 = Vec3::from_array(self.vertices[i0].position);
            let p1 = Vec3::from_array(self.vertices[i1].position);
            let p2 = Vec3::from_array(self.vertices[i2].position);

            let uv0 = Vec2::from_array(self.vertices[i0].uv);
            let uv1 = Vec2::from_array(self.vertices[i1].uv);
            let uv2 = Vec2::from_array(self.vertices[i2].uv);

            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let duv1 = uv1 - uv0;
            let duv2 = uv2 - uv0;

            let r = 1.0 / (duv1.x * duv2.y - duv2.x * duv1.y).max(1e-8);
            let tangent = (e1 * duv2.y - e2 * duv1.y) * r;
            let bitangent = (e2 * duv1.x - e1 * duv2.x) * r;

            tan1[i0] += tangent;
            tan1[i1] += tangent;
            tan1[i2] += tangent;

            tan2[i0] += bitangent;
            tan2[i1] += bitangent;
            tan2[i2] += bitangent;
        }

        for i in 0..self.vertices.len() {
            let n = Vec3::from_array(self.vertices[i].normal);
            let t = tan1[i];

            // Gram-Schmidt orthogonalise.
            let tangent = (t - n * n.dot(t)).normalize_or_zero();

            // Handedness.
            let w = if n.cross(t).dot(tan2[i]) < 0.0 { -1.0 } else { 1.0 };
            self.vertices[i].tangent = [tangent.x, tangent.y, tangent.z, w];
        }
    }
}

// ---------------------------------------------------------------------------
// MeshBuilder
// ---------------------------------------------------------------------------

/// Programmatic mesh construction API.
pub struct MeshBuilder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    pub fn new() -> Self { Self { vertices: Vec::new(), indices: Vec::new() } }

    /// Add a vertex, returning its index.
    pub fn add_vertex(&mut self, position: Vec3, normal: Vec3, uv: Vec2) -> u32 {
        let idx = self.vertices.len() as u32;
        self.vertices.push(Vertex::new(position, normal, Vec4::new(1.0, 0.0, 0.0, 1.0), uv));
        idx
    }

    /// Add a vertex with tangent.
    pub fn add_vertex_full(&mut self, position: Vec3, normal: Vec3, tangent: Vec4, uv: Vec2) -> u32 {
        let idx = self.vertices.len() as u32;
        self.vertices.push(Vertex::new(position, normal, tangent, uv));
        idx
    }

    /// Add a triangle by vertex indices.
    pub fn add_triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.push(a);
        self.indices.push(b);
        self.indices.push(c);
    }

    /// Add a quad as two triangles.
    pub fn add_quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.add_triangle(a, b, c);
        self.add_triangle(a, c, d);
    }

    /// Build the mesh, computing tangents automatically.
    pub fn build(self) -> Mesh {
        let mut mesh = Mesh::new(self.vertices, self.indices);
        mesh.compute_tangents();
        mesh
    }

    /// Build without computing tangents.
    pub fn build_no_tangents(self) -> Mesh {
        Mesh::new(self.vertices, self.indices)
    }

    /// Current vertex count.
    pub fn vertex_count(&self) -> u32 { self.vertices.len() as u32 }

    /// Current index count.
    pub fn index_count(&self) -> u32 { self.indices.len() as u32 }
}

impl Default for MeshBuilder {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Primitive generation
// ---------------------------------------------------------------------------

/// Create a unit cube mesh centred at the origin.
pub fn create_cube() -> Mesh {
    let mut builder = MeshBuilder::new();

    // 6 faces, each with 4 vertices and 2 triangles.
    let faces: [(Vec3, Vec3, Vec3); 6] = [
        (Vec3::Z, Vec3::X, Vec3::Y),     // +Z face
        (-Vec3::Z, -Vec3::X, Vec3::Y),   // -Z face
        (Vec3::X, -Vec3::Z, Vec3::Y),    // +X face
        (-Vec3::X, Vec3::Z, Vec3::Y),    // -X face
        (Vec3::Y, Vec3::X, -Vec3::Z),    // +Y face
        (-Vec3::Y, Vec3::X, Vec3::Z),    // -Y face
    ];

    for (normal, right, up) in &faces {
        let half = 0.5;
        let p0 = *normal * half - *right * half - *up * half;
        let p1 = *normal * half + *right * half - *up * half;
        let p2 = *normal * half + *right * half + *up * half;
        let p3 = *normal * half - *right * half + *up * half;

        let v0 = builder.add_vertex(p0, *normal, Vec2::new(0.0, 1.0));
        let v1 = builder.add_vertex(p1, *normal, Vec2::new(1.0, 1.0));
        let v2 = builder.add_vertex(p2, *normal, Vec2::new(1.0, 0.0));
        let v3 = builder.add_vertex(p3, *normal, Vec2::new(0.0, 0.0));

        builder.add_quad(v0, v1, v2, v3);
    }

    builder.build()
}

/// Create a UV sphere mesh.
pub fn create_sphere(segments: u32, rings: u32) -> Mesh {
    let mut builder = MeshBuilder::new();
    let segments = segments.max(3);
    let rings = rings.max(2);

    for ring in 0..=rings {
        let theta = PI * ring as f32 / rings as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for seg in 0..=segments {
            let phi = 2.0 * PI * seg as f32 / segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let x = sin_theta * cos_phi;
            let y = cos_theta;
            let z = sin_theta * sin_phi;

            let position = Vec3::new(x, y, z) * 0.5;
            let normal = Vec3::new(x, y, z);
            let uv = Vec2::new(seg as f32 / segments as f32, ring as f32 / rings as f32);

            builder.add_vertex(position, normal, uv);
        }
    }

    for ring in 0..rings {
        for seg in 0..segments {
            let current = ring * (segments + 1) + seg;
            let next = current + segments + 1;
            builder.add_triangle(current, next, current + 1);
            builder.add_triangle(current + 1, next, next + 1);
        }
    }

    builder.build()
}

/// Create a cylinder mesh (along Y axis).
pub fn create_cylinder(segments: u32, height: f32, radius: f32) -> Mesh {
    let mut builder = MeshBuilder::new();
    let segments = segments.max(3);
    let half_h = height * 0.5;

    // Side vertices.
    for i in 0..=segments {
        let phi = 2.0 * PI * i as f32 / segments as f32;
        let x = phi.cos() * radius;
        let z = phi.sin() * radius;
        let u = i as f32 / segments as f32;
        let normal = Vec3::new(phi.cos(), 0.0, phi.sin());

        builder.add_vertex(Vec3::new(x, -half_h, z), normal, Vec2::new(u, 1.0));
        builder.add_vertex(Vec3::new(x, half_h, z), normal, Vec2::new(u, 0.0));
    }

    // Side indices.
    for i in 0..segments {
        let base = i * 2;
        builder.add_triangle(base, base + 1, base + 2);
        builder.add_triangle(base + 2, base + 1, base + 3);
    }

    // Top cap.
    let top_center = builder.add_vertex(Vec3::new(0.0, half_h, 0.0), Vec3::Y, Vec2::new(0.5, 0.5));
    for i in 0..segments {
        let phi0 = 2.0 * PI * i as f32 / segments as f32;
        let phi1 = 2.0 * PI * (i + 1) as f32 / segments as f32;
        let v0 = builder.add_vertex(
            Vec3::new(phi0.cos() * radius, half_h, phi0.sin() * radius),
            Vec3::Y, Vec2::new(phi0.cos() * 0.5 + 0.5, phi0.sin() * 0.5 + 0.5),
        );
        let v1 = builder.add_vertex(
            Vec3::new(phi1.cos() * radius, half_h, phi1.sin() * radius),
            Vec3::Y, Vec2::new(phi1.cos() * 0.5 + 0.5, phi1.sin() * 0.5 + 0.5),
        );
        builder.add_triangle(top_center, v0, v1);
    }

    // Bottom cap.
    let bot_center = builder.add_vertex(Vec3::new(0.0, -half_h, 0.0), -Vec3::Y, Vec2::new(0.5, 0.5));
    for i in 0..segments {
        let phi0 = 2.0 * PI * i as f32 / segments as f32;
        let phi1 = 2.0 * PI * (i + 1) as f32 / segments as f32;
        let v0 = builder.add_vertex(
            Vec3::new(phi0.cos() * radius, -half_h, phi0.sin() * radius),
            -Vec3::Y, Vec2::new(phi0.cos() * 0.5 + 0.5, phi0.sin() * 0.5 + 0.5),
        );
        let v1 = builder.add_vertex(
            Vec3::new(phi1.cos() * radius, -half_h, phi1.sin() * radius),
            -Vec3::Y, Vec2::new(phi1.cos() * 0.5 + 0.5, phi1.sin() * 0.5 + 0.5),
        );
        builder.add_triangle(bot_center, v1, v0);
    }

    builder.build()
}

/// Create a plane mesh on the XZ plane.
pub fn create_plane(width: f32, depth: f32, subdivisions_x: u32, subdivisions_z: u32) -> Mesh {
    let mut builder = MeshBuilder::new();
    let sub_x = subdivisions_x.max(1);
    let sub_z = subdivisions_z.max(1);
    let half_w = width * 0.5;
    let half_d = depth * 0.5;

    for z in 0..=sub_z {
        for x in 0..=sub_x {
            let px = -half_w + width * x as f32 / sub_x as f32;
            let pz = -half_d + depth * z as f32 / sub_z as f32;
            let u = x as f32 / sub_x as f32;
            let v = z as f32 / sub_z as f32;
            builder.add_vertex(Vec3::new(px, 0.0, pz), Vec3::Y, Vec2::new(u, v));
        }
    }

    for z in 0..sub_z {
        for x in 0..sub_x {
            let tl = z * (sub_x + 1) + x;
            let tr = tl + 1;
            let bl = tl + sub_x + 1;
            let br = bl + 1;
            builder.add_quad(tl, bl, br, tr);
        }
    }

    builder.build()
}

/// Create a torus mesh.
pub fn create_torus(major_segments: u32, minor_segments: u32, major_radius: f32, minor_radius: f32) -> Mesh {
    let mut builder = MeshBuilder::new();
    let maj = major_segments.max(3);
    let min = minor_segments.max(3);

    for i in 0..=maj {
        let theta = 2.0 * PI * i as f32 / maj as f32;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let center = Vec3::new(cos_theta * major_radius, 0.0, sin_theta * major_radius);

        for j in 0..=min {
            let phi = 2.0 * PI * j as f32 / min as f32;
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();

            let x = (major_radius + minor_radius * cos_phi) * cos_theta;
            let y = minor_radius * sin_phi;
            let z = (major_radius + minor_radius * cos_phi) * sin_theta;

            let position = Vec3::new(x, y, z);
            let normal = (position - center).normalize_or_zero();
            let uv = Vec2::new(i as f32 / maj as f32, j as f32 / min as f32);

            builder.add_vertex(position, normal, uv);
        }
    }

    for i in 0..maj {
        for j in 0..min {
            let current = i * (min + 1) + j;
            let next = current + min + 1;
            builder.add_triangle(current, next, current + 1);
            builder.add_triangle(current + 1, next, next + 1);
        }
    }

    builder.build()
}

/// Create a capsule mesh (cylinder + two hemispheres).
pub fn create_capsule(segments: u32, rings_per_hemisphere: u32, radius: f32, height: f32) -> Mesh {
    let mut builder = MeshBuilder::new();
    let segments = segments.max(3);
    let rings = rings_per_hemisphere.max(1);
    let half_h = height * 0.5;

    // Top hemisphere.
    for ring in 0..=rings {
        let theta = (PI / 2.0) * ring as f32 / rings as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let y_offset = half_h + radius * cos_theta;

        for seg in 0..=segments {
            let phi = 2.0 * PI * seg as f32 / segments as f32;
            let x = sin_theta * phi.cos() * radius;
            let z = sin_theta * phi.sin() * radius;
            let position = Vec3::new(x, y_offset, z);
            let normal = Vec3::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());
            let uv = Vec2::new(seg as f32 / segments as f32, ring as f32 / (rings * 2 + 2) as f32);
            builder.add_vertex(position, normal, uv);
        }
    }

    // Cylinder body.
    for ring in 0..=1 {
        let y = if ring == 0 { half_h } else { -half_h };
        for seg in 0..=segments {
            let phi = 2.0 * PI * seg as f32 / segments as f32;
            let x = phi.cos() * radius;
            let z = phi.sin() * radius;
            let normal = Vec3::new(phi.cos(), 0.0, phi.sin());
            let v_coord = (rings as f32 + ring as f32) / (rings * 2 + 2) as f32;
            builder.add_vertex(Vec3::new(x, y, z), normal, Vec2::new(seg as f32 / segments as f32, v_coord));
        }
    }

    // Bottom hemisphere.
    for ring in 0..=rings {
        let theta = (PI / 2.0) + (PI / 2.0) * ring as f32 / rings as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let y_offset = -half_h + radius * cos_theta;

        for seg in 0..=segments {
            let phi = 2.0 * PI * seg as f32 / segments as f32;
            let x = sin_theta * phi.cos() * radius;
            let z = sin_theta * phi.sin() * radius;
            let position = Vec3::new(x, y_offset, z);
            let normal = Vec3::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());
            let v_coord = (rings as f32 + 2.0 + ring as f32) / (rings * 2 + 2) as f32;
            builder.add_vertex(position, normal, Vec2::new(seg as f32 / segments as f32, v_coord));
        }
    }

    // Generate indices for all rows.
    let total_rows = rings + 1 + 1 + rings; // top hemi + 2 cylinder rows + bottom hemi
    for row in 0..total_rows {
        for seg in 0..segments {
            let current = row * (segments + 1) + seg;
            let next = current + segments + 1;
            builder.add_triangle(current, next, current + 1);
            builder.add_triangle(current + 1, next, next + 1);
        }
    }

    builder.build()
}

/// Create a cone mesh.
pub fn create_cone(segments: u32, height: f32, radius: f32) -> Mesh {
    let mut builder = MeshBuilder::new();
    let segments = segments.max(3);
    let half_h = height * 0.5;

    // Apex vertex.
    let apex = builder.add_vertex(Vec3::new(0.0, half_h, 0.0), Vec3::Y, Vec2::new(0.5, 0.0));

    // Side vertices and triangles.
    let slope = (radius / height).atan();
    let cos_slope = slope.cos();
    let sin_slope = slope.sin();

    let mut side_ring = Vec::new();
    for i in 0..=segments {
        let phi = 2.0 * PI * i as f32 / segments as f32;
        let x = phi.cos() * radius;
        let z = phi.sin() * radius;
        let normal = Vec3::new(phi.cos() * cos_slope, sin_slope, phi.sin() * cos_slope).normalize_or_zero();
        let v = builder.add_vertex(Vec3::new(x, -half_h, z), normal, Vec2::new(i as f32 / segments as f32, 1.0));
        side_ring.push(v);
    }

    for i in 0..segments as usize {
        builder.add_triangle(apex, side_ring[i], side_ring[i + 1]);
    }

    // Bottom cap.
    let bot_center = builder.add_vertex(Vec3::new(0.0, -half_h, 0.0), -Vec3::Y, Vec2::new(0.5, 0.5));
    for i in 0..segments {
        let phi0 = 2.0 * PI * i as f32 / segments as f32;
        let phi1 = 2.0 * PI * (i + 1) as f32 / segments as f32;
        let v0 = builder.add_vertex(
            Vec3::new(phi0.cos() * radius, -half_h, phi0.sin() * radius),
            -Vec3::Y, Vec2::new(phi0.cos() * 0.5 + 0.5, phi0.sin() * 0.5 + 0.5),
        );
        let v1 = builder.add_vertex(
            Vec3::new(phi1.cos() * radius, -half_h, phi1.sin() * radius),
            -Vec3::Y, Vec2::new(phi1.cos() * 0.5 + 0.5, phi1.sin() * 0.5 + 0.5),
        );
        builder.add_triangle(bot_center, v1, v0);
    }

    builder.build()
}

/// Create a simple quad on the XY plane.
pub fn create_quad() -> Mesh {
    let mut builder = MeshBuilder::new();
    let v0 = builder.add_vertex(Vec3::new(-0.5, -0.5, 0.0), Vec3::Z, Vec2::new(0.0, 1.0));
    let v1 = builder.add_vertex(Vec3::new(0.5, -0.5, 0.0), Vec3::Z, Vec2::new(1.0, 1.0));
    let v2 = builder.add_vertex(Vec3::new(0.5, 0.5, 0.0), Vec3::Z, Vec2::new(1.0, 0.0));
    let v3 = builder.add_vertex(Vec3::new(-0.5, 0.5, 0.0), Vec3::Z, Vec2::new(0.0, 0.0));
    builder.add_quad(v0, v1, v2, v3);
    builder.build()
}

// ---------------------------------------------------------------------------
// MeshLOD
// ---------------------------------------------------------------------------

/// A set of mesh LOD (Level of Detail) variants.
pub struct MeshLOD {
    /// LOD levels, from highest detail (index 0) to lowest.
    pub levels: Vec<Mesh>,
    /// Screen-space area thresholds for LOD transitions.
    /// `thresholds[i]` is the minimum screen area for LOD level `i`.
    pub thresholds: Vec<f32>,
}

impl MeshLOD {
    /// Create a LOD set with a single level (no LOD).
    pub fn single(mesh: Mesh) -> Self {
        Self { levels: vec![mesh], thresholds: vec![0.0] }
    }

    /// Create a LOD set from multiple levels with uniform thresholds.
    pub fn from_levels(levels: Vec<Mesh>) -> Self {
        let n = levels.len();
        let thresholds: Vec<f32> = (0..n)
            .map(|i| {
                if i == 0 { 0.0 } else {
                    let t = i as f32 / n as f32;
                    t * t * 100.0 // quadratic falloff thresholds
                }
            })
            .collect();
        Self { levels, thresholds }
    }

    /// Number of LOD levels.
    pub fn level_count(&self) -> usize { self.levels.len() }

    /// Get the mesh for a given LOD level.
    pub fn get_level(&self, level: usize) -> Option<&Mesh> { self.levels.get(level) }
}

/// LOD level selector based on screen-space area.
pub struct LODSelector {
    /// Bias to prefer higher or lower LOD levels.
    pub bias: f32,
}

impl LODSelector {
    pub fn new() -> Self { Self { bias: 0.0 } }

    /// Select the appropriate LOD level based on screen-space projected area.
    ///
    /// # Arguments
    /// - `lod` -- the LOD set.
    /// - `screen_area` -- projected screen-space area in pixels^2.
    ///
    /// # Returns
    /// Index of the LOD level to use.
    pub fn select(&self, lod: &MeshLOD, screen_area: f32) -> usize {
        let biased_area = screen_area * (1.0 + self.bias);
        for (i, threshold) in lod.thresholds.iter().enumerate().rev() {
            if biased_area >= *threshold {
                return i;
            }
        }
        lod.levels.len() - 1
    }

    /// Compute screen-space projected area for a sphere.
    ///
    /// # Arguments
    /// - `center` -- world-space center of the bounding sphere.
    /// - `radius` -- radius of the bounding sphere.
    /// - `view_projection` -- combined view-projection matrix.
    /// - `viewport_height` -- viewport height in pixels.
    pub fn compute_screen_area(
        center: Vec3,
        radius: f32,
        view_projection: &glam::Mat4,
        viewport_height: f32,
    ) -> f32 {
        let center_clip = *view_projection * center.extend(1.0);
        if center_clip.w <= 0.0 { return 0.0; }
        let projected_radius = radius * viewport_height / center_clip.w;
        PI * projected_radius * projected_radius
    }
}

impl Default for LODSelector {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_format_standard() {
        let fmt = VertexFormat::standard();
        assert_eq!(fmt.attributes.len(), 4);
        assert!(fmt.has_attribute(VertexAttribute::Position));
        assert!(fmt.has_attribute(VertexAttribute::Normal));
        assert!(fmt.has_attribute(VertexAttribute::Tangent));
        assert!(fmt.has_attribute(VertexAttribute::UV0));
        assert!(!fmt.has_attribute(VertexAttribute::Color));
        assert_eq!(fmt.stride, 12 + 12 + 16 + 8);
    }

    #[test]
    fn vertex_format_offsets() {
        let fmt = VertexFormat::standard();
        assert_eq!(fmt.attribute_offset(VertexAttribute::Position), Some(0));
        assert_eq!(fmt.attribute_offset(VertexAttribute::Normal), Some(12));
        assert_eq!(fmt.attribute_offset(VertexAttribute::Tangent), Some(24));
        assert_eq!(fmt.attribute_offset(VertexAttribute::UV0), Some(40));
    }

    #[test]
    fn aabb_expand() {
        let mut aabb = AABB::default();
        aabb.expand_point(Vec3::ZERO);
        aabb.expand_point(Vec3::ONE);
        assert_eq!(aabb.min, Vec3::ZERO);
        assert_eq!(aabb.max, Vec3::ONE);
        assert!((aabb.center() - Vec3::splat(0.5)).length() < 1e-5);
    }

    #[test]
    fn aabb_intersection() {
        let a = AABB::new(Vec3::ZERO, Vec3::ONE);
        let b = AABB::new(Vec3::splat(0.5), Vec3::splat(1.5));
        let c = AABB::new(Vec3::splat(2.0), Vec3::splat(3.0));
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn mesh_builder_basic() {
        let mut builder = MeshBuilder::new();
        let v0 = builder.add_vertex(Vec3::ZERO, Vec3::Y, Vec2::ZERO);
        let v1 = builder.add_vertex(Vec3::X, Vec3::Y, Vec2::X);
        let v2 = builder.add_vertex(Vec3::Z, Vec3::Y, Vec2::Y);
        builder.add_triangle(v0, v1, v2);
        let mesh = builder.build();
        assert_eq!(mesh.vertex_count, 3);
        assert_eq!(mesh.index_count, 3);
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn create_cube_mesh() {
        let mesh = create_cube();
        assert_eq!(mesh.vertex_count, 24); // 6 faces * 4 vertices
        assert_eq!(mesh.triangle_count(), 12); // 6 faces * 2 triangles
        assert!(mesh.bounds.min.x <= -0.49);
        assert!(mesh.bounds.max.x >= 0.49);
    }

    #[test]
    fn create_sphere_mesh() {
        let mesh = create_sphere(16, 12);
        assert!(mesh.vertex_count > 0);
        assert!(mesh.triangle_count() > 0);
        // Sphere should fit in roughly [-0.5, 0.5].
        assert!(mesh.bounds.max.x <= 0.51);
        assert!(mesh.bounds.min.x >= -0.51);
    }

    #[test]
    fn create_cylinder_mesh() {
        let mesh = create_cylinder(16, 2.0, 0.5);
        assert!(mesh.vertex_count > 0);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn create_plane_mesh() {
        let mesh = create_plane(10.0, 10.0, 4, 4);
        assert_eq!(mesh.vertex_count, 5 * 5);
        assert_eq!(mesh.triangle_count(), 4 * 4 * 2);
    }

    #[test]
    fn create_torus_mesh() {
        let mesh = create_torus(16, 8, 1.0, 0.3);
        assert!(mesh.vertex_count > 0);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn create_capsule_mesh() {
        let mesh = create_capsule(12, 4, 0.5, 1.0);
        assert!(mesh.vertex_count > 0);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn create_cone_mesh() {
        let mesh = create_cone(12, 2.0, 0.5);
        assert!(mesh.vertex_count > 0);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn create_quad_mesh() {
        let mesh = create_quad();
        assert_eq!(mesh.vertex_count, 4);
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn mesh_compute_flat_normals() {
        let mut mesh = create_quad();
        mesh.compute_flat_normals();
        // All normals on a quad should point the same direction (+Z).
        for v in &mesh.vertices {
            let n = Vec3::from_array(v.normal);
            assert!((n.z - 1.0).abs() < 0.01 || (n.z + 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn mesh_bytes() {
        let mesh = create_cube();
        let vb = mesh.vertex_bytes();
        assert_eq!(vb.len(), mesh.vertex_count as usize * Vertex::SIZE);
        let ib = mesh.index_bytes();
        assert_eq!(ib.len(), mesh.index_count as usize * 4);
    }

    #[test]
    fn lod_selection() {
        let mesh0 = create_sphere(32, 24);
        let mesh1 = create_sphere(16, 12);
        let mesh2 = create_sphere(8, 6);

        let mut lod = MeshLOD::from_levels(vec![mesh0, mesh1, mesh2]);
        lod.thresholds = vec![0.0, 50.0, 200.0];

        let selector = LODSelector::new();
        assert_eq!(selector.select(&lod, 300.0), 2); // highest detail available above 200
        assert_eq!(selector.select(&lod, 100.0), 1); // medium
        assert_eq!(selector.select(&lod, 10.0), 0);  // lowest
    }

    #[test]
    fn screen_area_computation() {
        let vp = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let area = LODSelector::compute_screen_area(
            Vec3::new(0.0, 0.0, -5.0), 1.0, &vp, 1080.0,
        );
        assert!(area > 0.0);
    }
}
