// engine/render/src/environment_probe.rs
//
// Environment probe system: capture cubemaps at probe positions for indirect
// lighting, with parallax correction (box/sphere projection), probe blending
// by distance, real-time probe update scheduling, and SH (spherical harmonics)
// projection for diffuse indirect lighting.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn normalize(self) -> Self { let l = self.length(); if l < 1e-12 { Self::ZERO } else { Self { x:self.x/l, y:self.y/l, z:self.z/l } } }
    pub fn scale(self, s: f32) -> Self { Self { x:self.x*s, y:self.y*s, z:self.z*s } }
    pub fn add(self, r: Self) -> Self { Self { x:self.x+r.x, y:self.y+r.y, z:self.z+r.z } }
    pub fn sub(self, r: Self) -> Self { Self { x:self.x-r.x, y:self.y-r.y, z:self.z-r.z } }
    pub fn mul_elem(self, r: Self) -> Self { Self { x:self.x*r.x, y:self.y*r.y, z:self.z*r.z } }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
    pub fn abs(self) -> Self { Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() } }
    pub fn max_component(self) -> f32 { self.x.max(self.y).max(self.z) }
    pub fn min_component(self) -> f32 { self.x.min(self.y).min(self.z) }
}

// ---------------------------------------------------------------------------
// Axis-Aligned Bounding Box
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn center(&self) -> Vec3 {
        self.min.add(self.max).scale(0.5)
    }

    pub fn extents(&self) -> Vec3 {
        self.max.sub(self.min).scale(0.5)
    }

    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x && point.x <= self.max.x
        && point.y >= self.min.y && point.y <= self.max.y
        && point.z >= self.min.z && point.z <= self.max.z
    }

    /// Compute the closest point on the AABB surface to a given point.
    pub fn closest_point(&self, point: Vec3) -> Vec3 {
        Vec3::new(
            point.x.clamp(self.min.x, self.max.x),
            point.y.clamp(self.min.y, self.max.y),
            point.z.clamp(self.min.z, self.max.z),
        )
    }

    /// Ray-AABB intersection, returns (t_enter, t_exit).
    pub fn ray_intersect(&self, origin: Vec3, dir: Vec3) -> Option<(f32, f32)> {
        let inv_dir = Vec3::new(
            if dir.x.abs() > 1e-12 { 1.0 / dir.x } else { f32::MAX },
            if dir.y.abs() > 1e-12 { 1.0 / dir.y } else { f32::MAX },
            if dir.z.abs() > 1e-12 { 1.0 / dir.z } else { f32::MAX },
        );

        let t1 = (self.min.x - origin.x) * inv_dir.x;
        let t2 = (self.max.x - origin.x) * inv_dir.x;
        let t3 = (self.min.y - origin.y) * inv_dir.y;
        let t4 = (self.max.y - origin.y) * inv_dir.y;
        let t5 = (self.min.z - origin.z) * inv_dir.z;
        let t6 = (self.max.z - origin.z) * inv_dir.z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if tmax < 0.0 || tmin > tmax {
            None
        } else {
            Some((tmin, tmax))
        }
    }
}

// ---------------------------------------------------------------------------
// Cubemap face directions
// ---------------------------------------------------------------------------

/// The six faces of a cubemap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubemapFace {
    PositiveX = 0,
    NegativeX = 1,
    PositiveY = 2,
    NegativeY = 3,
    PositiveZ = 4,
    NegativeZ = 5,
}

impl CubemapFace {
    pub const ALL: [Self; 6] = [
        Self::PositiveX, Self::NegativeX,
        Self::PositiveY, Self::NegativeY,
        Self::PositiveZ, Self::NegativeZ,
    ];

    /// Get the view direction for this cubemap face.
    pub fn direction(&self) -> Vec3 {
        match self {
            Self::PositiveX => Vec3::new(1.0, 0.0, 0.0),
            Self::NegativeX => Vec3::new(-1.0, 0.0, 0.0),
            Self::PositiveY => Vec3::new(0.0, 1.0, 0.0),
            Self::NegativeY => Vec3::new(0.0, -1.0, 0.0),
            Self::PositiveZ => Vec3::new(0.0, 0.0, 1.0),
            Self::NegativeZ => Vec3::new(0.0, 0.0, -1.0),
        }
    }

    /// Get the up vector for this cubemap face.
    pub fn up(&self) -> Vec3 {
        match self {
            Self::PositiveX => Vec3::new(0.0, 1.0, 0.0),
            Self::NegativeX => Vec3::new(0.0, 1.0, 0.0),
            Self::PositiveY => Vec3::new(0.0, 0.0, -1.0),
            Self::NegativeY => Vec3::new(0.0, 0.0, 1.0),
            Self::PositiveZ => Vec3::new(0.0, 1.0, 0.0),
            Self::NegativeZ => Vec3::new(0.0, 1.0, 0.0),
        }
    }
}

/// Convert a 3D direction to cubemap face + UV.
pub fn direction_to_cubemap(dir: Vec3) -> (CubemapFace, f32, f32) {
    let abs_dir = dir.abs();
    let (face, ma, sc, tc) = if abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z {
        if dir.x > 0.0 {
            (CubemapFace::PositiveX, abs_dir.x, -dir.z, -dir.y)
        } else {
            (CubemapFace::NegativeX, abs_dir.x, dir.z, -dir.y)
        }
    } else if abs_dir.y >= abs_dir.x && abs_dir.y >= abs_dir.z {
        if dir.y > 0.0 {
            (CubemapFace::PositiveY, abs_dir.y, dir.x, dir.z)
        } else {
            (CubemapFace::NegativeY, abs_dir.y, dir.x, -dir.z)
        }
    } else {
        if dir.z > 0.0 {
            (CubemapFace::PositiveZ, abs_dir.z, dir.x, -dir.y)
        } else {
            (CubemapFace::NegativeZ, abs_dir.z, -dir.x, -dir.y)
        }
    };

    let u = 0.5 * (sc / ma + 1.0);
    let v = 0.5 * (tc / ma + 1.0);
    (face, u, v)
}

// ---------------------------------------------------------------------------
// Spherical Harmonics (L2)
// ---------------------------------------------------------------------------

/// Order-2 spherical harmonics (9 coefficients) for diffuse irradiance.
#[derive(Debug, Clone)]
pub struct SphericalHarmonicsL2 {
    pub coefficients: [Vec3; 9],
}

impl Default for SphericalHarmonicsL2 {
    fn default() -> Self {
        Self { coefficients: [Vec3::ZERO; 9] }
    }
}

impl SphericalHarmonicsL2 {
    /// Evaluate the SH at a direction to get irradiance.
    pub fn evaluate(&self, dir: Vec3) -> Vec3 {
        let c = &self.coefficients;
        // SH basis functions
        let y0 = 0.282095;     // l=0, m=0
        let y1 = 0.488603;     // l=1
        let y2_0 = 1.092548;   // l=2, |m|=1
        let y2_1 = 0.315392;   // l=2, m=0
        let y2_2 = 0.546274;   // l=2, |m|=2

        let result = c[0].scale(y0)
            .add(c[1].scale(y1 * dir.y))
            .add(c[2].scale(y1 * dir.z))
            .add(c[3].scale(y1 * dir.x))
            .add(c[4].scale(y2_0 * dir.x * dir.y))
            .add(c[5].scale(y2_0 * dir.y * dir.z))
            .add(c[6].scale(y2_1 * (3.0 * dir.z * dir.z - 1.0)))
            .add(c[7].scale(y2_0 * dir.x * dir.z))
            .add(c[8].scale(y2_2 * (dir.x * dir.x - dir.y * dir.y)));

        // Clamp to avoid negative light
        Vec3::new(result.x.max(0.0), result.y.max(0.0), result.z.max(0.0))
    }

    /// Project cubemap data into SH coefficients.
    pub fn project_cubemap(faces: &[Vec<Vec3>; 6], face_size: u32) -> Self {
        let mut sh = SphericalHarmonicsL2::default();
        let mut total_weight = 0.0_f32;

        for (face_idx, face_data) in faces.iter().enumerate() {
            let face = CubemapFace::ALL[face_idx];

            for y in 0..face_size {
                for x in 0..face_size {
                    let u = (x as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;

                    // Direction for this texel
                    let dir = cubemap_texel_direction(face, u, v);

                    // Solid angle weight (accounts for cubemap distortion)
                    let solid_angle = 4.0 / (face_size as f32 * face_size as f32
                        * (1.0 + u * u + v * v).powf(1.5));

                    let color = face_data[(y * face_size + x) as usize];
                    let weight = solid_angle;
                    total_weight += weight;

                    // Project into SH
                    let basis = sh_basis(dir);
                    for (i, &b) in basis.iter().enumerate() {
                        sh.coefficients[i] = sh.coefficients[i].add(color.scale(b * weight));
                    }
                }
            }
        }

        // Normalize
        if total_weight > 0.0 {
            let inv = 4.0 * PI / total_weight;
            for c in &mut sh.coefficients {
                *c = c.scale(inv);
            }
        }

        sh
    }

    /// Add another SH scaled by a weight.
    pub fn add_weighted(&mut self, other: &SphericalHarmonicsL2, weight: f32) {
        for i in 0..9 {
            self.coefficients[i] = self.coefficients[i].add(other.coefficients[i].scale(weight));
        }
    }

    /// Lerp between two SH.
    pub fn lerp(&self, other: &SphericalHarmonicsL2, t: f32) -> SphericalHarmonicsL2 {
        let mut result = SphericalHarmonicsL2::default();
        for i in 0..9 {
            result.coefficients[i] = self.coefficients[i].lerp(other.coefficients[i], t);
        }
        result
    }
}

fn cubemap_texel_direction(face: CubemapFace, u: f32, v: f32) -> Vec3 {
    let dir = match face {
        CubemapFace::PositiveX => Vec3::new(1.0, -v, -u),
        CubemapFace::NegativeX => Vec3::new(-1.0, -v, u),
        CubemapFace::PositiveY => Vec3::new(u, 1.0, v),
        CubemapFace::NegativeY => Vec3::new(u, -1.0, -v),
        CubemapFace::PositiveZ => Vec3::new(u, -v, 1.0),
        CubemapFace::NegativeZ => Vec3::new(-u, -v, -1.0),
    };
    dir.normalize()
}

fn sh_basis(dir: Vec3) -> [f32; 9] {
    let (x, y, z) = (dir.x, dir.y, dir.z);
    [
        0.282095,                    // l=0, m=0
        0.488603 * y,                // l=1, m=-1
        0.488603 * z,                // l=1, m=0
        0.488603 * x,                // l=1, m=1
        1.092548 * x * y,            // l=2, m=-2
        1.092548 * y * z,            // l=2, m=-1
        0.315392 * (3.0*z*z - 1.0),  // l=2, m=0
        1.092548 * x * z,            // l=2, m=1
        0.546274 * (x*x - y*y),      // l=2, m=2
    ]
}

// ---------------------------------------------------------------------------
// Environment probe
// ---------------------------------------------------------------------------

/// Unique identifier for a probe.
pub type ProbeId = u32;

/// Projection shape for parallax correction.
#[derive(Debug, Clone, Copy)]
pub enum ProbeShape {
    /// Box-shaped influence zone.
    Box { half_extents: Vec3 },
    /// Sphere-shaped influence zone.
    Sphere { radius: f32 },
}

/// An environment probe captures a cubemap for reflections.
#[derive(Debug, Clone)]
pub struct EnvironmentProbe {
    pub id: ProbeId,
    pub position: Vec3,
    pub shape: ProbeShape,
    pub influence_radius: f32,
    pub blend_distance: f32,   // fade-out region at the edge
    pub priority: i32,         // higher = more important
    pub cubemap_resolution: u32,
    pub is_realtime: bool,     // true = update every frame
    pub update_interval: f32,  // seconds between updates (for non-realtime)
    pub time_since_update: f32,
    pub is_dirty: bool,

    // Captured data
    pub cubemap_faces: [Vec<Vec3>; 6],
    pub irradiance_sh: SphericalHarmonicsL2,
    pub prefiltered_mip_count: u32,
}

impl EnvironmentProbe {
    pub fn new(id: ProbeId, position: Vec3, shape: ProbeShape) -> Self {
        let influence_radius = match shape {
            ProbeShape::Box { half_extents } => half_extents.max_component(),
            ProbeShape::Sphere { radius } => radius,
        };

        Self {
            id,
            position,
            shape,
            influence_radius,
            blend_distance: 1.0,
            priority: 0,
            cubemap_resolution: 128,
            is_realtime: false,
            update_interval: 5.0,
            time_since_update: f32::MAX,
            is_dirty: true,
            cubemap_faces: Default::default(),
            irradiance_sh: SphericalHarmonicsL2::default(),
            prefiltered_mip_count: 5,
        }
    }

    /// Compute the blend weight for this probe at a given world position.
    pub fn blend_weight(&self, sample_pos: Vec3) -> f32 {
        match self.shape {
            ProbeShape::Box { half_extents } => {
                let local = sample_pos.sub(self.position);
                let abs_local = local.abs();

                // Distance to the nearest face of the box
                let dx = (half_extents.x - abs_local.x).max(0.0);
                let dy = (half_extents.y - abs_local.y).max(0.0);
                let dz = (half_extents.z - abs_local.z).max(0.0);

                // Outside the box
                if abs_local.x > half_extents.x || abs_local.y > half_extents.y || abs_local.z > half_extents.z {
                    return 0.0;
                }

                let min_dist = dx.min(dy).min(dz);
                (min_dist / self.blend_distance.max(0.01)).clamp(0.0, 1.0)
            }

            ProbeShape::Sphere { radius } => {
                let dist = sample_pos.distance(self.position);
                if dist > radius {
                    return 0.0;
                }
                let edge_dist = radius - dist;
                (edge_dist / self.blend_distance.max(0.01)).clamp(0.0, 1.0)
            }
        }
    }

    /// Parallax-correct a reflection direction for box projection.
    pub fn parallax_correct_box(&self, sample_pos: Vec3, reflection: Vec3) -> Vec3 {
        match self.shape {
            ProbeShape::Box { half_extents } => {
                let aabb = AABB::new(
                    self.position.sub(half_extents),
                    self.position.add(half_extents),
                );

                // Intersect reflection ray with AABB
                if let Some((_, t_exit)) = aabb.ray_intersect(sample_pos, reflection) {
                    let hit = sample_pos.add(reflection.scale(t_exit));
                    hit.sub(self.position).normalize()
                } else {
                    reflection
                }
            }
            ProbeShape::Sphere { radius } => {
                // Sphere parallax correction
                let offset = sample_pos.sub(self.position);
                let a = reflection.dot(reflection);
                let b = 2.0 * offset.dot(reflection);
                let c = offset.dot(offset) - radius * radius;
                let discriminant = b * b - 4.0 * a * c;

                if discriminant >= 0.0 {
                    let t = (-b + discriminant.sqrt()) / (2.0 * a);
                    let hit = sample_pos.add(reflection.scale(t));
                    hit.sub(self.position).normalize()
                } else {
                    reflection
                }
            }
        }
    }

    /// Check if this probe needs an update.
    pub fn needs_update(&self) -> bool {
        if self.is_dirty {
            return true;
        }
        if self.is_realtime {
            return true;
        }
        self.time_since_update >= self.update_interval
    }

    /// Initialize cubemap faces with the given resolution.
    pub fn initialize_cubemap(&mut self) {
        let res = self.cubemap_resolution;
        let size = (res * res) as usize;
        for face in &mut self.cubemap_faces {
            face.resize(size, Vec3::ZERO);
        }
    }

    /// After cubemap capture, project to SH for diffuse irradiance.
    pub fn compute_irradiance_sh(&mut self) {
        self.irradiance_sh = SphericalHarmonicsL2::project_cubemap(
            &self.cubemap_faces,
            self.cubemap_resolution,
        );
    }

    /// Generate prefiltered mip levels for specular IBL.
    pub fn prefilter_specular(&self, roughness: f32) -> Vec<Vec3> {
        let size = (self.cubemap_resolution >> 1).max(1);
        let total = (size * size) as usize;
        let mut result = vec![Vec3::ZERO; total * 6];

        for face_idx in 0..6 {
            let face = CubemapFace::ALL[face_idx];

            for y in 0..size {
                for x in 0..size {
                    let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;

                    let n = cubemap_texel_direction(face, u, v);

                    // Importance-sample around the normal
                    let filtered = self.prefilter_sample(n, roughness, 64);
                    let out_idx = face_idx * total + (y * size + x) as usize;
                    result[out_idx] = filtered;
                }
            }
        }

        result
    }

    fn prefilter_sample(&self, normal: Vec3, roughness: f32, num_samples: u32) -> Vec3 {
        let mut total_color = Vec3::ZERO;
        let mut total_weight = 0.0_f32;

        for i in 0..num_samples {
            let xi = hammersley_2d(i, num_samples);
            let h = importance_sample_ggx(xi, normal, roughness);
            let l = h.scale(2.0 * normal.dot(h)).sub(normal);

            let n_dot_l = normal.dot(l);
            if n_dot_l > 0.0 {
                let (face, fu, fv) = direction_to_cubemap(l);
                let face_data = &self.cubemap_faces[face as usize];
                let res = self.cubemap_resolution;
                let px = ((fu * res as f32) as u32).min(res - 1);
                let py = ((fv * res as f32) as u32).min(res - 1);
                let idx = (py * res + px) as usize;

                if idx < face_data.len() {
                    total_color = total_color.add(face_data[idx].scale(n_dot_l));
                    total_weight += n_dot_l;
                }
            }
        }

        if total_weight > 0.0 {
            total_color.scale(1.0 / total_weight)
        } else {
            Vec3::ZERO
        }
    }
}

fn hammersley_2d(i: u32, n: u32) -> [f32; 2] {
    let mut bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    [i as f32 / n as f32, bits as f32 * 2.328_306_4e-10]
}

fn importance_sample_ggx(xi: [f32; 2], n: Vec3, roughness: f32) -> Vec3 {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    let h = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);

    let up = if n.z.abs() < 0.999 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let tangent = up.cross(n).normalize();
    let bitangent = n.cross(tangent);

    tangent.scale(h.x).add(bitangent.scale(h.y)).add(n.scale(h.z)).normalize()
}

// Implement Default for arrays of Vec<Vec3>
impl Default for Vec<Vec3> {
    // We can't impl Default for Vec<Vec3>, so we handle this differently
}

// ---------------------------------------------------------------------------
// Probe manager
// ---------------------------------------------------------------------------

/// Manages all environment probes in the scene.
pub struct EnvironmentProbeManager {
    probes: Vec<EnvironmentProbe>,
    next_id: ProbeId,
    pub max_active_probes: usize,
    pub max_updates_per_frame: u32,
    pub max_faces_per_frame: u32, // for spreading capture across frames
    pub default_fallback_sh: SphericalHarmonicsL2,

    // Update scheduling
    update_queue: Vec<ProbeId>,
    current_update_face: u32,
    current_update_probe: Option<ProbeId>,
}

impl EnvironmentProbeManager {
    pub fn new() -> Self {
        Self {
            probes: Vec::new(),
            next_id: 1,
            max_active_probes: 16,
            max_updates_per_frame: 1,
            max_faces_per_frame: 1,
            default_fallback_sh: SphericalHarmonicsL2::default(),
            update_queue: Vec::new(),
            current_update_face: 0,
            current_update_probe: None,
        }
    }

    /// Add a new probe.
    pub fn add_probe(&mut self, position: Vec3, shape: ProbeShape) -> ProbeId {
        let id = self.next_id;
        self.next_id += 1;
        let mut probe = EnvironmentProbe::new(id, position, shape);
        probe.initialize_cubemap();
        self.probes.push(probe);
        id
    }

    /// Remove a probe by ID.
    pub fn remove_probe(&mut self, id: ProbeId) -> bool {
        if let Some(idx) = self.probes.iter().position(|p| p.id == id) {
            self.probes.remove(idx);
            true
        } else {
            false
        }
    }

    /// Get a probe by ID.
    pub fn get_probe(&self, id: ProbeId) -> Option<&EnvironmentProbe> {
        self.probes.iter().find(|p| p.id == id)
    }

    /// Get a mutable probe by ID.
    pub fn get_probe_mut(&mut self, id: ProbeId) -> Option<&mut EnvironmentProbe> {
        self.probes.iter_mut().find(|p| p.id == id)
    }

    /// Find the N most relevant probes for a given world position.
    pub fn find_affecting_probes(&self, position: Vec3, max_probes: usize) -> Vec<ProbeContribution> {
        let mut contributions = Vec::new();

        for probe in &self.probes {
            let weight = probe.blend_weight(position);
            if weight > 0.0 {
                contributions.push(ProbeContribution {
                    probe_id: probe.id,
                    weight,
                    distance: position.distance(probe.position),
                    priority: probe.priority,
                });
            }
        }

        // Sort by priority (higher first), then by distance (closer first)
        contributions.sort_by(|a, b| {
            b.priority.cmp(&a.priority)
                .then_with(|| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))
        });

        contributions.truncate(max_probes);

        // Normalize weights
        let total_weight: f32 = contributions.iter().map(|c| c.weight).sum();
        if total_weight > 0.0 {
            for c in &mut contributions {
                c.weight /= total_weight;
            }
        }

        contributions
    }

    /// Evaluate indirect diffuse lighting at a position using probes.
    pub fn evaluate_diffuse(&self, position: Vec3, normal: Vec3) -> Vec3 {
        let contributions = self.find_affecting_probes(position, 4);

        if contributions.is_empty() {
            return self.default_fallback_sh.evaluate(normal);
        }

        let mut result = Vec3::ZERO;
        for contrib in &contributions {
            if let Some(probe) = self.get_probe(contrib.probe_id) {
                let irradiance = probe.irradiance_sh.evaluate(normal);
                result = result.add(irradiance.scale(contrib.weight));
            }
        }

        result
    }

    /// Evaluate indirect specular reflection at a position using probes.
    pub fn evaluate_specular(
        &self,
        position: Vec3,
        reflection: Vec3,
        roughness: f32,
    ) -> Vec3 {
        let contributions = self.find_affecting_probes(position, 2);

        if contributions.is_empty() {
            return Vec3::ZERO;
        }

        let mut result = Vec3::ZERO;
        for contrib in &contributions {
            if let Some(probe) = self.get_probe(contrib.probe_id) {
                // Parallax-correct the reflection direction
                let corrected = probe.parallax_correct_box(position, reflection);

                // Sample the cubemap (simplified - would use prefiltered mip in practice)
                let (face, u, v) = direction_to_cubemap(corrected);
                let face_data = &probe.cubemap_faces[face as usize];
                let res = probe.cubemap_resolution;
                let px = ((u * res as f32) as u32).min(res - 1);
                let py = ((v * res as f32) as u32).min(res - 1);
                let idx = (py * res + px) as usize;

                if idx < face_data.len() {
                    result = result.add(face_data[idx].scale(contrib.weight));
                }
            }
        }

        result
    }

    /// Update probe scheduling. Call once per frame.
    pub fn update(&mut self, dt: f32, camera_position: Vec3) {
        // Update time trackers
        for probe in &mut self.probes {
            probe.time_since_update += dt;
        }

        // Build update queue
        self.update_queue.clear();
        for probe in &self.probes {
            if probe.needs_update() {
                self.update_queue.push(probe.id);
            }
        }

        // Prioritize: dirty first, then by distance to camera, then by time since update
        let probes_ref = &self.probes;
        self.update_queue.sort_by(|&a, &b| {
            let pa = probes_ref.iter().find(|p| p.id == a);
            let pb = probes_ref.iter().find(|p| p.id == b);
            match (pa, pb) {
                (Some(pa), Some(pb)) => {
                    // Dirty first
                    pb.is_dirty.cmp(&pa.is_dirty)
                        // Closer to camera first
                        .then_with(|| {
                            let da = pa.position.distance(camera_position);
                            let db = pb.position.distance(camera_position);
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        // Longer since update first
                        .then_with(|| {
                            pb.time_since_update.partial_cmp(&pa.time_since_update)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                }
                _ => std::cmp::Ordering::Equal,
            }
        });
    }

    /// Get the next face to render for probe updates.
    /// Returns None if no probes need updating this frame.
    pub fn next_capture_task(&mut self) -> Option<CaptureTask> {
        if self.current_update_probe.is_none() {
            self.current_update_probe = self.update_queue.first().copied();
            self.current_update_face = 0;
        }

        if let Some(probe_id) = self.current_update_probe {
            if let Some(probe) = self.probes.iter().find(|p| p.id == probe_id) {
                if self.current_update_face < 6 {
                    let face = CubemapFace::ALL[self.current_update_face as usize];
                    let task = CaptureTask {
                        probe_id,
                        face,
                        position: probe.position,
                        resolution: probe.cubemap_resolution,
                    };

                    self.current_update_face += 1;

                    if self.current_update_face >= 6 {
                        // All faces done, mark probe as updated
                        if let Some(p) = self.probes.iter_mut().find(|p| p.id == probe_id) {
                            p.is_dirty = false;
                            p.time_since_update = 0.0;
                            p.compute_irradiance_sh();
                        }
                        self.current_update_probe = None;
                        self.update_queue.retain(|&id| id != probe_id);
                    }

                    return Some(task);
                }
            }
        }

        None
    }

    /// Get statistics.
    pub fn stats(&self) -> ProbeStats {
        ProbeStats {
            total_probes: self.probes.len(),
            dirty_probes: self.probes.iter().filter(|p| p.is_dirty).count(),
            realtime_probes: self.probes.iter().filter(|p| p.is_realtime).count(),
            pending_updates: self.update_queue.len(),
        }
    }
}

/// A capture task for rendering one face of a probe cubemap.
#[derive(Debug, Clone)]
pub struct CaptureTask {
    pub probe_id: ProbeId,
    pub face: CubemapFace,
    pub position: Vec3,
    pub resolution: u32,
}

#[derive(Debug, Clone)]
pub struct ProbeContribution {
    pub probe_id: ProbeId,
    pub weight: f32,
    pub distance: f32,
    pub priority: i32,
}

#[derive(Debug, Clone)]
pub struct ProbeStats {
    pub total_probes: usize,
    pub dirty_probes: usize,
    pub realtime_probes: usize,
    pub pending_updates: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_blend_sphere() {
        let probe = EnvironmentProbe::new(
            1,
            Vec3::ZERO,
            ProbeShape::Sphere { radius: 10.0 },
        );
        assert!(probe.blend_weight(Vec3::ZERO) > 0.0);
        assert_eq!(probe.blend_weight(Vec3::new(20.0, 0.0, 0.0)), 0.0);
    }

    #[test]
    fn test_probe_blend_box() {
        let probe = EnvironmentProbe::new(
            1,
            Vec3::ZERO,
            ProbeShape::Box { half_extents: Vec3::new(5.0, 5.0, 5.0) },
        );
        assert!(probe.blend_weight(Vec3::ZERO) > 0.0);
        assert_eq!(probe.blend_weight(Vec3::new(10.0, 0.0, 0.0)), 0.0);
    }

    #[test]
    fn test_direction_to_cubemap() {
        let (face, u, v) = direction_to_cubemap(Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(face, CubemapFace::PositiveX);
        assert!((u - 0.5).abs() < 0.01);
        assert!((v - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_probe_manager() {
        let mut manager = EnvironmentProbeManager::new();
        let id = manager.add_probe(Vec3::ZERO, ProbeShape::Sphere { radius: 10.0 });
        assert!(manager.get_probe(id).is_some());

        let contribs = manager.find_affecting_probes(Vec3::new(1.0, 0.0, 0.0), 4);
        assert_eq!(contribs.len(), 1);
    }

    #[test]
    fn test_sh_evaluate() {
        let sh = SphericalHarmonicsL2::default();
        let result = sh.evaluate(Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(result.x, 0.0);
    }

    #[test]
    fn test_aabb_ray_intersect() {
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let result = aabb.ray_intersect(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(result.is_some());
    }
}
