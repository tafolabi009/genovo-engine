// engine/render/src/gpu_particles.rs
//
// GPU-computed particle simulation for the Genovo engine.
//
// All particle data lives entirely on the GPU. CPU only configures emitters,
// dispatches compute passes, and issues indirect draw calls. Features:
//
// - Compute-shader particle update: integrate velocity, apply forces, age, kill
// - Compute-shader particle emit: initialize new particles from emitter shape
// - Indirect draw: particle count from atomic counter
// - Bitonic sort by view-depth for correct alpha blending
// - WGSL compute shader source embedded as string constants

use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// WGSL Shader Sources
// ---------------------------------------------------------------------------

/// WGSL compute shader for the particle update pass.
///
/// Each invocation processes one particle: integrates velocity, applies forces
/// (gravity, wind, curl noise, attractor), ages the particle, and kills expired
/// particles by swapping with the tail of the alive list.
pub const PARTICLE_UPDATE_WGSL: &str = r#"
// ---------------------------------------------------------------------------
// Particle update compute shader
// ---------------------------------------------------------------------------

struct Particle {
    position: vec3<f32>,
    lifetime: f32,
    velocity: vec3<f32>,
    age: f32,
    color: vec4<f32>,
    size: vec2<f32>,
    rotation: f32,
    angular_velocity: f32,
};

struct EmitterParams {
    position: vec3<f32>,
    emit_rate: f32,
    direction: vec3<f32>,
    spread_angle: f32,
    min_speed: f32,
    max_speed: f32,
    min_lifetime: f32,
    max_lifetime: f32,
    start_color: vec4<f32>,
    end_color: vec4<f32>,
    start_size: vec2<f32>,
    end_size: vec2<f32>,
    gravity: vec3<f32>,
    _pad0: f32,
    wind: vec3<f32>,
    drag: f32,
    noise_strength: f32,
    noise_frequency: f32,
    noise_scroll_speed: f32,
    _pad1: f32,
    attractor_position: vec3<f32>,
    attractor_strength: f32,
    shape_type: u32,       // 0=point, 1=sphere, 2=box, 3=cone
    shape_radius: f32,
    shape_extent_x: f32,
    shape_extent_y: f32,
    shape_extent_z: f32,
    floor_y: f32,
    bounce_factor: f32,
    _pad2: f32,
};

struct SimUniforms {
    delta_time: f32,
    total_time: f32,
    max_particles: u32,
    _pad: u32,
    view_matrix: mat4x4<f32>,
};

struct AtomicCounter {
    alive_count: atomic<u32>,
    dead_count: atomic<u32>,
    emit_count: u32,
    draw_vertex_count: u32,
    draw_instance_count: atomic<u32>,
    draw_first_vertex: u32,
    draw_first_instance: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> alive_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> dead_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> counter: AtomicCounter;
@group(0) @binding(4) var<uniform> emitter: EmitterParams;
@group(0) @binding(5) var<uniform> sim: SimUniforms;
@group(0) @binding(6) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> sort_values: array<u32>;

// Simple 3D hash for curl noise approximation
fn hash3(p: vec3<f32>) -> vec3<f32> {
    var q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6))
    );
    return fract(sin(q) * 43758.5453123) * 2.0 - 1.0;
}

// Approximate curl noise for turbulence
fn curl_noise(p: vec3<f32>) -> vec3<f32> {
    let e = 0.01;
    let px = hash3(p + vec3<f32>(e, 0.0, 0.0)) - hash3(p - vec3<f32>(e, 0.0, 0.0));
    let py = hash3(p + vec3<f32>(0.0, e, 0.0)) - hash3(p - vec3<f32>(0.0, e, 0.0));
    let pz = hash3(p + vec3<f32>(0.0, 0.0, e)) - hash3(p - vec3<f32>(0.0, 0.0, e));
    return vec3<f32>(
        py.z - pz.y,
        pz.x - px.z,
        px.y - py.x
    ) / (2.0 * e);
}

@compute @workgroup_size(256)
fn update_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let alive_count = atomicLoad(&counter.alive_count);
    if gid.x >= alive_count {
        return;
    }

    let index = alive_indices[gid.x];
    var p = particles[index];

    // Age the particle
    p.age += sim.delta_time;
    if p.age >= p.lifetime {
        // Kill: push index onto dead list
        let dead_slot = atomicAdd(&counter.dead_count, 1u);
        dead_indices[dead_slot] = index;
        // Mark sort key as max so it sorts to the end
        sort_keys[gid.x] = 0xFFFFFFFFu;
        particles[index] = p;
        return;
    }

    let life_t = p.age / p.lifetime;

    // --- Forces ---

    // Gravity
    p.velocity += emitter.gravity * sim.delta_time;

    // Wind
    p.velocity += emitter.wind * sim.delta_time;

    // Drag (exponential decay)
    p.velocity *= exp(-emitter.drag * sim.delta_time);

    // Curl noise turbulence
    if emitter.noise_strength > 0.0 {
        let noise_pos = p.position * emitter.noise_frequency
            + vec3<f32>(0.0, sim.total_time * emitter.noise_scroll_speed, 0.0);
        let curl = curl_noise(noise_pos);
        p.velocity += curl * emitter.noise_strength * sim.delta_time;
    }

    // Attractor
    if emitter.attractor_strength != 0.0 {
        let to_attractor = emitter.attractor_position - p.position;
        let dist_sq = max(dot(to_attractor, to_attractor), 0.01);
        let dir = normalize(to_attractor);
        p.velocity += dir * emitter.attractor_strength / dist_sq * sim.delta_time;
    }

    // --- Integrate position ---
    p.position += p.velocity * sim.delta_time;

    // Floor collision / bounce
    if p.position.y < emitter.floor_y && emitter.bounce_factor > 0.0 {
        p.position.y = emitter.floor_y;
        p.velocity.y = -p.velocity.y * emitter.bounce_factor;
        p.velocity.x *= 0.9;
        p.velocity.z *= 0.9;
    }

    // --- Interpolate visual properties ---
    p.color = mix(emitter.start_color, emitter.end_color, life_t);
    p.size = mix(emitter.start_size, emitter.end_size, life_t);
    p.rotation += p.angular_velocity * sim.delta_time;

    particles[index] = p;

    // Mark alive in draw indirect
    atomicAdd(&counter.draw_instance_count, 1u);

    // Compute view-space depth for sorting
    let view_pos = sim.view_matrix * vec4<f32>(p.position, 1.0);
    let depth = -view_pos.z;
    // Encode depth as u32 for bitonic sort (flip sign bit for correct ordering)
    let depth_bits = bitcast<u32>(depth);
    let sort_key = select(depth_bits ^ 0x7FFFFFFFu, depth_bits, (depth_bits & 0x80000000u) != 0u);
    sort_keys[gid.x] = sort_key;
    sort_values[gid.x] = index;
}
"#;

/// WGSL compute shader for the particle emit pass.
///
/// Dispatched with `ceil(emit_count / 256)` workgroups. Each invocation
/// pops a dead index and initializes a new particle based on emitter shape.
pub const PARTICLE_EMIT_WGSL: &str = r#"
// ---------------------------------------------------------------------------
// Particle emit compute shader
// ---------------------------------------------------------------------------

struct Particle {
    position: vec3<f32>,
    lifetime: f32,
    velocity: vec3<f32>,
    age: f32,
    color: vec4<f32>,
    size: vec2<f32>,
    rotation: f32,
    angular_velocity: f32,
};

struct EmitterParams {
    position: vec3<f32>,
    emit_rate: f32,
    direction: vec3<f32>,
    spread_angle: f32,
    min_speed: f32,
    max_speed: f32,
    min_lifetime: f32,
    max_lifetime: f32,
    start_color: vec4<f32>,
    end_color: vec4<f32>,
    start_size: vec2<f32>,
    end_size: vec2<f32>,
    gravity: vec3<f32>,
    _pad0: f32,
    wind: vec3<f32>,
    drag: f32,
    noise_strength: f32,
    noise_frequency: f32,
    noise_scroll_speed: f32,
    _pad1: f32,
    attractor_position: vec3<f32>,
    attractor_strength: f32,
    shape_type: u32,
    shape_radius: f32,
    shape_extent_x: f32,
    shape_extent_y: f32,
    shape_extent_z: f32,
    floor_y: f32,
    bounce_factor: f32,
    _pad2: f32,
};

struct SimUniforms {
    delta_time: f32,
    total_time: f32,
    max_particles: u32,
    _pad: u32,
    view_matrix: mat4x4<f32>,
};

struct AtomicCounter {
    alive_count: atomic<u32>,
    dead_count: atomic<u32>,
    emit_count: u32,
    draw_vertex_count: u32,
    draw_instance_count: atomic<u32>,
    draw_first_vertex: u32,
    draw_first_instance: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> alive_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> dead_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> counter: AtomicCounter;
@group(0) @binding(4) var<uniform> emitter: EmitterParams;
@group(0) @binding(5) var<uniform> sim: SimUniforms;

// PCG random number generator
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_float(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967295.0;
}

fn rand_float2(seed: u32) -> vec2<f32> {
    return vec2<f32>(rand_float(seed), rand_float(seed + 1u));
}

fn rand_float3(seed: u32) -> vec3<f32> {
    return vec3<f32>(rand_float(seed), rand_float(seed + 1u), rand_float(seed + 2u));
}

// Sample point on unit sphere
fn random_on_sphere(seed: u32) -> vec3<f32> {
    let u = rand_float(seed) * 2.0 - 1.0;
    let theta = rand_float(seed + 7u) * 6.28318530718;
    let r = sqrt(1.0 - u * u);
    return vec3<f32>(r * cos(theta), r * sin(theta), u);
}

// Sample direction within cone around `dir` with half-angle `angle`
fn random_in_cone(dir: vec3<f32>, angle: f32, seed: u32) -> vec3<f32> {
    let cos_angle = cos(angle);
    let z = rand_float(seed) * (1.0 - cos_angle) + cos_angle;
    let phi = rand_float(seed + 3u) * 6.28318530718;
    let r = sqrt(1.0 - z * z);
    let local = vec3<f32>(r * cos(phi), r * sin(phi), z);

    // Build tangent frame from `dir`
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(dot(dir, up)) > 0.99 {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t = normalize(cross(up, dir));
    let b = cross(dir, t);

    return t * local.x + b * local.y + dir * local.z;
}

@compute @workgroup_size(256)
fn emit_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= counter.emit_count {
        return;
    }

    // Try to pop a dead index
    let dead_slot = atomicSub(&counter.dead_count, 1u);
    if dead_slot == 0u {
        // No dead particles available, undo the decrement
        atomicAdd(&counter.dead_count, 1u);
        return;
    }
    let index = dead_indices[dead_slot - 1u];

    let seed_base = gid.x * 17u + bitcast<u32>(sim.total_time * 1000.0);

    // Determine spawn position based on emitter shape
    var spawn_offset = vec3<f32>(0.0);
    switch emitter.shape_type {
        case 0u: {
            // Point: no offset
        }
        case 1u: {
            // Sphere: random point inside sphere
            let dir = random_on_sphere(seed_base + 100u);
            let r = pow(rand_float(seed_base + 200u), 1.0 / 3.0) * emitter.shape_radius;
            spawn_offset = dir * r;
        }
        case 2u: {
            // Box: random point inside box
            spawn_offset = vec3<f32>(
                (rand_float(seed_base + 300u) * 2.0 - 1.0) * emitter.shape_extent_x,
                (rand_float(seed_base + 301u) * 2.0 - 1.0) * emitter.shape_extent_y,
                (rand_float(seed_base + 302u) * 2.0 - 1.0) * emitter.shape_extent_z,
            );
        }
        case 3u: {
            // Cone base: random point on disk at cone base
            let angle = rand_float(seed_base + 400u) * 6.28318530718;
            let r = sqrt(rand_float(seed_base + 401u)) * emitter.shape_radius;
            spawn_offset = vec3<f32>(r * cos(angle), 0.0, r * sin(angle));
        }
        default: {
        }
    }

    var p: Particle;
    p.position = emitter.position + spawn_offset;
    p.age = 0.0;
    p.lifetime = mix(emitter.min_lifetime, emitter.max_lifetime, rand_float(seed_base + 500u));
    p.color = emitter.start_color;
    p.size = emitter.start_size;
    p.rotation = rand_float(seed_base + 600u) * 6.28318530718;
    p.angular_velocity = (rand_float(seed_base + 700u) * 2.0 - 1.0) * 3.14159;

    // Initial velocity
    let speed = mix(emitter.min_speed, emitter.max_speed, rand_float(seed_base + 800u));
    let dir = random_in_cone(emitter.direction, emitter.spread_angle, seed_base + 900u);
    p.velocity = dir * speed;

    particles[index] = p;

    // Push onto alive list
    let alive_slot = atomicAdd(&counter.alive_count, 1u);
    alive_indices[alive_slot] = index;
}
"#;

/// WGSL compute shader for bitonic merge sort.
///
/// Sorts particles by view-space depth for correct back-to-front alpha blending.
/// The sort operates on (key, value) pairs where key = encoded depth,
/// value = particle index.
pub const PARTICLE_SORT_WGSL: &str = r#"
// ---------------------------------------------------------------------------
// Bitonic sort compute shader (one pass per sub-sequence)
// ---------------------------------------------------------------------------

struct SortUniforms {
    block_size: u32,
    sub_block_size: u32,
    count: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> values: array<u32>;
@group(0) @binding(2) var<uniform> params: SortUniforms;

@compute @workgroup_size(256)
fn bitonic_sort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count {
        return;
    }

    let block = params.block_size;
    let sub = params.sub_block_size;

    // Determine partner index for this step
    let half_sub = sub >> 1u;
    let group = i / sub;
    let pos = i % sub;
    let ascending = (group % 2u) == 0u;

    var partner: u32;
    if pos < half_sub {
        partner = i + half_sub;
    } else {
        partner = i - half_sub;
    }

    if partner >= params.count {
        return;
    }

    let key_i = keys[i];
    let key_p = keys[partner];

    let should_swap_asc = pos < half_sub && key_i > key_p;
    let should_swap_desc = pos < half_sub && key_i < key_p;
    let should_swap = select(should_swap_desc, should_swap_asc, ascending);

    if should_swap {
        keys[i] = key_p;
        keys[partner] = key_i;
        let val_i = values[i];
        let val_p = values[partner];
        values[i] = val_p;
        values[partner] = val_i;
    }
}
"#;

// ---------------------------------------------------------------------------
// EmitterShape
// ---------------------------------------------------------------------------

/// Shape from which particles are spawned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmitterShape {
    /// Particles spawn at a single point.
    Point,
    /// Particles spawn within a sphere of the given radius.
    Sphere { radius: f32 },
    /// Particles spawn within an axis-aligned box.
    Box { half_extents: Vec3 },
    /// Particles spawn from the base of a cone (disk).
    Cone { radius: f32 },
}

impl EmitterShape {
    /// Returns the shape type index matching the WGSL `shape_type` field.
    pub fn type_index(&self) -> u32 {
        match self {
            EmitterShape::Point => 0,
            EmitterShape::Sphere { .. } => 1,
            EmitterShape::Box { .. } => 2,
            EmitterShape::Cone { .. } => 3,
        }
    }
}

impl Default for EmitterShape {
    fn default() -> Self {
        EmitterShape::Point
    }
}

// ---------------------------------------------------------------------------
// GpuParticleSettings
// ---------------------------------------------------------------------------

/// Configuration for a GPU particle system.
#[derive(Debug, Clone)]
pub struct GpuParticleSettings {
    /// Maximum number of particles (determines GPU buffer sizes).
    pub max_particles: u32,
    /// Particles emitted per second.
    pub emit_rate: f32,
    /// Minimum particle lifetime in seconds.
    pub min_lifetime: f32,
    /// Maximum particle lifetime in seconds.
    pub max_lifetime: f32,
    /// Minimum initial speed.
    pub min_speed: f32,
    /// Maximum initial speed.
    pub max_speed: f32,
    /// Direction around which initial velocity is spread.
    pub direction: Vec3,
    /// Half-angle of the emission cone in radians.
    pub spread_angle: f32,
    /// Gravity acceleration applied to all particles.
    pub gravity: Vec3,
    /// Wind force applied each frame.
    pub wind: Vec3,
    /// Drag coefficient (exponential velocity decay).
    pub drag: f32,
    /// Curl noise turbulence strength.
    pub noise_strength: f32,
    /// Curl noise sample frequency (world-space scale).
    pub noise_frequency: f32,
    /// Speed at which the noise field scrolls (vertical).
    pub noise_scroll_speed: f32,
    /// Attractor point position.
    pub attractor_position: Vec3,
    /// Attractor strength (negative = repulsor).
    pub attractor_strength: f32,
    /// Emitter shape.
    pub shape: EmitterShape,
    /// Particle start color (RGBA).
    pub start_color: Vec4,
    /// Particle end color (RGBA) — interpolated over lifetime.
    pub end_color: Vec4,
    /// Particle start size (width, height).
    pub start_size: Vec2,
    /// Particle end size (width, height).
    pub end_size: Vec2,
    /// Y coordinate of collision floor. Particles bounce off this plane.
    pub floor_y: f32,
    /// Velocity preservation on floor bounce (0 = no bounce, 1 = perfect).
    pub bounce_factor: f32,
    /// Whether to sort particles by depth each frame (required for alpha).
    pub depth_sort: bool,
}

impl Default for GpuParticleSettings {
    fn default() -> Self {
        Self {
            max_particles: 65536,
            emit_rate: 1000.0,
            min_lifetime: 1.0,
            max_lifetime: 3.0,
            min_speed: 1.0,
            max_speed: 5.0,
            direction: Vec3::Y,
            spread_angle: std::f32::consts::FRAC_PI_6,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            wind: Vec3::ZERO,
            drag: 0.1,
            noise_strength: 0.0,
            noise_frequency: 1.0,
            noise_scroll_speed: 0.5,
            attractor_position: Vec3::ZERO,
            attractor_strength: 0.0,
            shape: EmitterShape::Point,
            start_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            end_color: Vec4::new(1.0, 1.0, 1.0, 0.0),
            start_size: Vec2::new(0.1, 0.1),
            end_size: Vec2::new(0.05, 0.05),
            floor_y: f32::NEG_INFINITY,
            bounce_factor: 0.0,
            depth_sort: true,
        }
    }
}

impl GpuParticleSettings {
    /// Creates settings for a fire-like effect.
    pub fn fire() -> Self {
        Self {
            max_particles: 10000,
            emit_rate: 500.0,
            min_lifetime: 0.5,
            max_lifetime: 1.5,
            min_speed: 2.0,
            max_speed: 5.0,
            direction: Vec3::Y,
            spread_angle: 0.3,
            gravity: Vec3::new(0.0, 2.0, 0.0), // Upward buoyancy
            wind: Vec3::ZERO,
            drag: 1.5,
            noise_strength: 3.0,
            noise_frequency: 2.0,
            noise_scroll_speed: 1.0,
            attractor_position: Vec3::ZERO,
            attractor_strength: 0.0,
            shape: EmitterShape::Sphere { radius: 0.3 },
            start_color: Vec4::new(1.0, 0.8, 0.2, 1.0),
            end_color: Vec4::new(1.0, 0.1, 0.0, 0.0),
            start_size: Vec2::splat(0.2),
            end_size: Vec2::splat(0.5),
            floor_y: f32::NEG_INFINITY,
            bounce_factor: 0.0,
            depth_sort: true,
        }
    }

    /// Creates settings for a rain-like effect.
    pub fn rain() -> Self {
        Self {
            max_particles: 50000,
            emit_rate: 5000.0,
            min_lifetime: 1.0,
            max_lifetime: 2.0,
            min_speed: 10.0,
            max_speed: 15.0,
            direction: Vec3::new(0.0, -1.0, 0.0),
            spread_angle: 0.05,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            wind: Vec3::new(1.0, 0.0, 0.5),
            drag: 0.01,
            noise_strength: 0.0,
            noise_frequency: 1.0,
            noise_scroll_speed: 0.0,
            attractor_position: Vec3::ZERO,
            attractor_strength: 0.0,
            shape: EmitterShape::Box {
                half_extents: Vec3::new(20.0, 0.1, 20.0),
            },
            start_color: Vec4::new(0.7, 0.8, 1.0, 0.6),
            end_color: Vec4::new(0.7, 0.8, 1.0, 0.0),
            start_size: Vec2::new(0.02, 0.15),
            end_size: Vec2::new(0.01, 0.1),
            floor_y: 0.0,
            bounce_factor: 0.0,
            depth_sort: false,
        }
    }

    /// Creates settings for a snow-like effect.
    pub fn snow() -> Self {
        Self {
            max_particles: 30000,
            emit_rate: 2000.0,
            min_lifetime: 4.0,
            max_lifetime: 8.0,
            min_speed: 0.5,
            max_speed: 1.5,
            direction: Vec3::new(0.0, -1.0, 0.0),
            spread_angle: 0.4,
            gravity: Vec3::new(0.0, -0.5, 0.0),
            wind: Vec3::new(0.5, 0.0, 0.3),
            drag: 2.0,
            noise_strength: 1.0,
            noise_frequency: 0.5,
            noise_scroll_speed: 0.2,
            attractor_position: Vec3::ZERO,
            attractor_strength: 0.0,
            shape: EmitterShape::Box {
                half_extents: Vec3::new(15.0, 0.1, 15.0),
            },
            start_color: Vec4::new(1.0, 1.0, 1.0, 0.9),
            end_color: Vec4::new(1.0, 1.0, 1.0, 0.0),
            start_size: Vec2::splat(0.05),
            end_size: Vec2::splat(0.03),
            floor_y: 0.0,
            bounce_factor: 0.0,
            depth_sort: true,
        }
    }

    /// Creates settings for a magical sparkle effect.
    pub fn sparkle() -> Self {
        Self {
            max_particles: 5000,
            emit_rate: 300.0,
            min_lifetime: 0.5,
            max_lifetime: 2.0,
            min_speed: 0.5,
            max_speed: 3.0,
            direction: Vec3::Y,
            spread_angle: std::f32::consts::PI,
            gravity: Vec3::new(0.0, -1.0, 0.0),
            wind: Vec3::ZERO,
            drag: 0.5,
            noise_strength: 2.0,
            noise_frequency: 3.0,
            noise_scroll_speed: 0.5,
            attractor_position: Vec3::ZERO,
            attractor_strength: 0.0,
            shape: EmitterShape::Sphere { radius: 0.5 },
            start_color: Vec4::new(0.5, 0.8, 1.0, 1.0),
            end_color: Vec4::new(1.0, 0.5, 1.0, 0.0),
            start_size: Vec2::splat(0.08),
            end_size: Vec2::splat(0.01),
            floor_y: f32::NEG_INFINITY,
            bounce_factor: 0.0,
            depth_sort: true,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuEmitter
// ---------------------------------------------------------------------------

/// An emitter that produces GPU particles.
///
/// The emitter's parameters are uploaded as a uniform buffer each frame.
/// Multiple emitters can reference the same [`GpuParticleSystem`] by sharing
/// its handle.
#[derive(Debug, Clone)]
pub struct GpuEmitter {
    /// World-space position of the emitter.
    pub position: Vec3,
    /// Whether the emitter is actively spawning particles.
    pub enabled: bool,
    /// Accumulated fractional particles (for sub-frame emit accuracy).
    emit_accumulator: f32,
    /// The settings that control this emitter's behavior.
    pub settings: GpuParticleSettings,
}

impl GpuEmitter {
    /// Creates a new emitter at the given position with the given settings.
    pub fn new(position: Vec3, settings: GpuParticleSettings) -> Self {
        Self {
            position,
            enabled: true,
            emit_accumulator: 0.0,
            settings,
        }
    }

    /// Computes how many particles to emit this frame and packs the emitter
    /// uniform data.
    pub fn prepare_frame(&mut self, dt: f32) -> EmitterUniformData {
        let emit_count = if self.enabled {
            self.emit_accumulator += self.settings.emit_rate * dt;
            let count = self.emit_accumulator as u32;
            self.emit_accumulator -= count as f32;
            count.min(self.settings.max_particles)
        } else {
            0
        };

        let (shape_radius, extent_x, extent_y, extent_z) = match self.settings.shape {
            EmitterShape::Point => (0.0, 0.0, 0.0, 0.0),
            EmitterShape::Sphere { radius } => (radius, 0.0, 0.0, 0.0),
            EmitterShape::Box { half_extents } => {
                (0.0, half_extents.x, half_extents.y, half_extents.z)
            }
            EmitterShape::Cone { radius } => (radius, 0.0, 0.0, 0.0),
        };

        EmitterUniformData {
            position: self.position,
            emit_rate: self.settings.emit_rate,
            direction: self.settings.direction.normalize_or_zero(),
            spread_angle: self.settings.spread_angle,
            min_speed: self.settings.min_speed,
            max_speed: self.settings.max_speed,
            min_lifetime: self.settings.min_lifetime,
            max_lifetime: self.settings.max_lifetime,
            start_color: self.settings.start_color,
            end_color: self.settings.end_color,
            start_size: self.settings.start_size,
            end_size: self.settings.end_size,
            gravity: self.settings.gravity,
            wind: self.settings.wind,
            drag: self.settings.drag,
            noise_strength: self.settings.noise_strength,
            noise_frequency: self.settings.noise_frequency,
            noise_scroll_speed: self.settings.noise_scroll_speed,
            attractor_position: self.settings.attractor_position,
            attractor_strength: self.settings.attractor_strength,
            shape_type: self.settings.shape.type_index(),
            shape_radius,
            shape_extent_x: extent_x,
            shape_extent_y: extent_y,
            shape_extent_z: extent_z,
            floor_y: self.settings.floor_y,
            bounce_factor: self.settings.bounce_factor,
            emit_count,
        }
    }

    /// Resets the emit accumulator (e.g. after teleporting the emitter).
    pub fn reset_accumulator(&mut self) {
        self.emit_accumulator = 0.0;
    }

    /// Sets a new position for the emitter.
    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }
}

// ---------------------------------------------------------------------------
// EmitterUniformData
// ---------------------------------------------------------------------------

/// Packed emitter data matching the WGSL `EmitterParams` struct layout.
///
/// This is uploaded to a GPU uniform buffer each frame.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmitterUniformData {
    pub position: Vec3,
    pub emit_rate: f32,
    pub direction: Vec3,
    pub spread_angle: f32,
    pub min_speed: f32,
    pub max_speed: f32,
    pub min_lifetime: f32,
    pub max_lifetime: f32,
    pub start_color: Vec4,
    pub end_color: Vec4,
    pub start_size: Vec2,
    pub end_size: Vec2,
    pub gravity: Vec3,
    // _pad0
    pub wind: Vec3,
    pub drag: f32,
    pub noise_strength: f32,
    pub noise_frequency: f32,
    pub noise_scroll_speed: f32,
    // _pad1
    pub attractor_position: Vec3,
    pub attractor_strength: f32,
    pub shape_type: u32,
    pub shape_radius: f32,
    pub shape_extent_x: f32,
    pub shape_extent_y: f32,
    pub shape_extent_z: f32,
    pub floor_y: f32,
    pub bounce_factor: f32,
    // _pad2
    pub emit_count: u32,
}

// ---------------------------------------------------------------------------
// SimUniformData
// ---------------------------------------------------------------------------

/// Packed simulation uniforms matching the WGSL `SimUniforms` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SimUniformData {
    pub delta_time: f32,
    pub total_time: f32,
    pub max_particles: u32,
    pub _pad: u32,
    pub view_matrix: Mat4,
}

// ---------------------------------------------------------------------------
// CounterData
// ---------------------------------------------------------------------------

/// CPU-side mirror of the GPU atomic counter buffer.
///
/// Used for initialising the buffer and reading back draw-indirect args.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CounterData {
    pub alive_count: u32,
    pub dead_count: u32,
    pub emit_count: u32,
    /// Number of vertices per instance (6 for a billboard quad = 2 triangles).
    pub draw_vertex_count: u32,
    /// Number of instances to draw (filled by update shader).
    pub draw_instance_count: u32,
    pub draw_first_vertex: u32,
    pub draw_first_instance: u32,
    pub _pad: u32,
}

impl CounterData {
    /// Creates initial counter data with all particles on the dead list.
    pub fn new_initial(max_particles: u32) -> Self {
        Self {
            alive_count: 0,
            dead_count: max_particles,
            emit_count: 0,
            draw_vertex_count: 6,
            draw_instance_count: 0,
            draw_first_vertex: 0,
            draw_first_instance: 0,
            _pad: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// BitonicSortPass
// ---------------------------------------------------------------------------

/// Tracks the state of a multi-dispatch bitonic merge sort.
///
/// Bitonic sort requires `O(log^2 N)` dispatches, each comparing and
/// swapping elements at different strides. This struct generates the
/// sequence of (block_size, sub_block_size) pairs.
#[derive(Debug, Clone)]
pub struct BitonicSortPass {
    /// Current block size (doubles each outer iteration).
    block_size: u32,
    /// Current sub-block size within the block (halves each inner iteration).
    sub_block_size: u32,
    /// Total number of elements to sort.
    count: u32,
    /// Whether the sequence has been exhausted.
    done: bool,
}

impl BitonicSortPass {
    /// Creates a new sort pass for `count` elements.
    pub fn new(count: u32) -> Self {
        // Round up to next power of two
        let n = count.next_power_of_two();
        Self {
            block_size: 2,
            sub_block_size: 2,
            count: n,
            done: n <= 1,
        }
    }

    /// Returns the next (block_size, sub_block_size, count) triple for a
    /// compute dispatch, or `None` if the sort is complete.
    pub fn next_pass(&mut self) -> Option<SortPassParams> {
        if self.done {
            return None;
        }

        let params = SortPassParams {
            block_size: self.block_size,
            sub_block_size: self.sub_block_size,
            count: self.count,
        };

        // Advance: halve sub_block_size; if it reaches 1, double block_size
        self.sub_block_size /= 2;
        if self.sub_block_size < 2 {
            self.block_size *= 2;
            if self.block_size > self.count {
                self.done = true;
            } else {
                self.sub_block_size = self.block_size;
            }
        }

        Some(params)
    }

    /// Resets the sort pass to start a new sort over `count` elements.
    pub fn reset(&mut self, count: u32) {
        let n = count.next_power_of_two();
        self.block_size = 2;
        self.sub_block_size = 2;
        self.count = n;
        self.done = n <= 1;
    }

    /// Returns the total number of dispatches required for a full sort.
    pub fn total_passes(count: u32) -> u32 {
        let n = count.next_power_of_two();
        if n <= 1 {
            return 0;
        }
        let log_n = (n as f32).log2() as u32;
        log_n * (log_n + 1) / 2
    }
}

/// Parameters for a single bitonic sort dispatch.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SortPassParams {
    pub block_size: u32,
    pub sub_block_size: u32,
    pub count: u32,
}

// ---------------------------------------------------------------------------
// GpuParticleSystem
// ---------------------------------------------------------------------------

/// Manages one or more GPU particle emitters and their shared GPU resources.
///
/// The system owns the particle buffer, alive/dead index lists, the atomic
/// counter buffer, and the sort buffers. Each frame:
///
/// 1. Reset the draw-indirect instance count to 0.
/// 2. For each emitter, dispatch the emit compute shader.
/// 3. Dispatch the update compute shader over all alive particles.
/// 4. (Optional) Run bitonic sort passes for depth-sorted alpha blending.
/// 5. Issue an indirect draw call using the counter buffer as indirect args.
#[derive(Debug)]
pub struct GpuParticleSystem {
    /// Unique name for this particle system (for debugging).
    pub name: String,
    /// Maximum number of particles across all emitters.
    pub max_particles: u32,
    /// Whether depth sorting is enabled.
    pub depth_sort_enabled: bool,
    /// Emitters attached to this system, keyed by name.
    emitters: HashMap<String, GpuEmitter>,
    /// Total elapsed simulation time.
    total_time: f32,
    /// Whether the system has been initialised on the GPU.
    initialized: bool,
    /// Whether the system is paused.
    paused: bool,
    /// Time scale multiplier (1.0 = normal speed).
    time_scale: f32,
    /// Warmup time: simulate this many seconds on first frame.
    warmup_time: f32,
    /// Whether warmup has been performed.
    warmup_done: bool,
}

impl GpuParticleSystem {
    /// Creates a new GPU particle system.
    pub fn new(name: impl Into<String>, max_particles: u32) -> Self {
        Self {
            name: name.into(),
            max_particles,
            depth_sort_enabled: true,
            emitters: HashMap::new(),
            total_time: 0.0,
            initialized: false,
            paused: false,
            time_scale: 1.0,
            warmup_time: 0.0,
            warmup_done: false,
        }
    }

    /// Adds an emitter to the system.
    pub fn add_emitter(&mut self, name: impl Into<String>, emitter: GpuEmitter) {
        self.emitters.insert(name.into(), emitter);
    }

    /// Removes an emitter by name.
    pub fn remove_emitter(&mut self, name: &str) -> Option<GpuEmitter> {
        self.emitters.remove(name)
    }

    /// Returns a reference to an emitter by name.
    pub fn emitter(&self, name: &str) -> Option<&GpuEmitter> {
        self.emitters.get(name)
    }

    /// Returns a mutable reference to an emitter by name.
    pub fn emitter_mut(&mut self, name: &str) -> Option<&mut GpuEmitter> {
        self.emitters.get_mut(name)
    }

    /// Returns an iterator over all emitters.
    pub fn emitters(&self) -> impl Iterator<Item = (&str, &GpuEmitter)> {
        self.emitters.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Sets the time scale multiplier.
    pub fn set_time_scale(&mut self, scale: f32) {
        self.time_scale = scale.max(0.0);
    }

    /// Sets a warmup time (seconds to pre-simulate on first frame).
    pub fn set_warmup_time(&mut self, seconds: f32) {
        self.warmup_time = seconds.max(0.0);
    }

    /// Pauses the simulation.
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resumes the simulation.
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Returns whether the system is paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Prepares per-frame data for all emitters and returns the list of
    /// compute dispatches needed.
    ///
    /// Returns `(sim_uniforms, emitter_uniforms, sort_needed)`.
    pub fn prepare_frame(
        &mut self,
        dt: f32,
        view_matrix: Mat4,
    ) -> GpuParticleFrameData {
        let effective_dt = if self.paused { 0.0 } else { dt * self.time_scale };

        // Handle warmup: simulate in fixed steps
        let warmup_steps = if !self.warmup_done && self.warmup_time > 0.0 {
            self.warmup_done = true;
            let step_dt = 1.0 / 60.0;
            let steps = (self.warmup_time / step_dt) as u32;
            Some((steps, step_dt))
        } else {
            None
        };

        self.total_time += effective_dt;

        let sim = SimUniformData {
            delta_time: effective_dt,
            total_time: self.total_time,
            max_particles: self.max_particles,
            _pad: 0,
            view_matrix,
        };

        let emitter_data: Vec<(String, EmitterUniformData)> = self
            .emitters
            .iter_mut()
            .map(|(name, emitter)| {
                let data = emitter.prepare_frame(effective_dt);
                (name.clone(), data)
            })
            .collect();

        GpuParticleFrameData {
            sim_uniforms: sim,
            emitter_uniforms: emitter_data,
            sort_needed: self.depth_sort_enabled,
            max_particles: self.max_particles,
            warmup_steps,
        }
    }

    /// Resets the entire system: clears all alive particles and resets
    /// emitter accumulators.
    pub fn reset(&mut self) {
        self.total_time = 0.0;
        self.initialized = false;
        self.warmup_done = false;
        for emitter in self.emitters.values_mut() {
            emitter.reset_accumulator();
        }
    }

    /// Returns the initial counter data for GPU buffer initialization.
    pub fn initial_counter_data(&self) -> CounterData {
        CounterData::new_initial(self.max_particles)
    }

    /// Returns the initial dead index list (all indices 0..max_particles).
    pub fn initial_dead_indices(&self) -> Vec<u32> {
        (0..self.max_particles).collect()
    }

    /// Computes the required buffer sizes in bytes.
    pub fn buffer_sizes(&self) -> GpuParticleBufferSizes {
        let particle_stride = 64; // sizeof(Particle) in WGSL: 16*4 bytes
        GpuParticleBufferSizes {
            particle_buffer: self.max_particles as u64 * particle_stride,
            alive_index_buffer: self.max_particles as u64 * 4,
            dead_index_buffer: self.max_particles as u64 * 4,
            counter_buffer: 32, // 8 u32 fields
            sort_key_buffer: self.max_particles as u64 * 4,
            sort_value_buffer: self.max_particles as u64 * 4,
        }
    }

    /// Returns how many workgroups to dispatch for update (256 threads each).
    pub fn update_workgroups(&self) -> u32 {
        (self.max_particles + 255) / 256
    }

    /// Returns how many workgroups to dispatch for a given emit count.
    pub fn emit_workgroups(emit_count: u32) -> u32 {
        (emit_count + 255) / 256
    }

    /// Returns how many workgroups for the sort shader.
    pub fn sort_workgroups(&self) -> u32 {
        let n = self.max_particles.next_power_of_two();
        (n + 255) / 256
    }
}

// ---------------------------------------------------------------------------
// GpuParticleFrameData
// ---------------------------------------------------------------------------

/// Data prepared by [`GpuParticleSystem::prepare_frame`] for GPU upload.
#[derive(Debug)]
pub struct GpuParticleFrameData {
    /// Simulation uniform data for the frame.
    pub sim_uniforms: SimUniformData,
    /// Per-emitter uniform data, keyed by emitter name.
    pub emitter_uniforms: Vec<(String, EmitterUniformData)>,
    /// Whether depth sorting should be performed.
    pub sort_needed: bool,
    /// Maximum particle count (for workgroup dispatch calculation).
    pub max_particles: u32,
    /// Warmup simulation: `Some((steps, dt_per_step))` on the first frame.
    pub warmup_steps: Option<(u32, f32)>,
}

// ---------------------------------------------------------------------------
// GpuParticleBufferSizes
// ---------------------------------------------------------------------------

/// Sizes (in bytes) of the GPU buffers required by a particle system.
#[derive(Debug, Clone, Copy)]
pub struct GpuParticleBufferSizes {
    /// Storage buffer holding all Particle structs.
    pub particle_buffer: u64,
    /// Storage buffer for alive particle indices.
    pub alive_index_buffer: u64,
    /// Storage buffer for dead (free) particle indices.
    pub dead_index_buffer: u64,
    /// Storage buffer for atomic counters and indirect draw args.
    pub counter_buffer: u64,
    /// Storage buffer for sort keys (depth values).
    pub sort_key_buffer: u64,
    /// Storage buffer for sort values (particle indices).
    pub sort_value_buffer: u64,
}

impl GpuParticleBufferSizes {
    /// Returns the total GPU memory required in bytes.
    pub fn total(&self) -> u64 {
        self.particle_buffer
            + self.alive_index_buffer
            + self.dead_index_buffer
            + self.counter_buffer
            + self.sort_key_buffer
            + self.sort_value_buffer
    }
}

// ---------------------------------------------------------------------------
// GpuParticleComponent
// ---------------------------------------------------------------------------

/// ECS component that attaches a GPU particle system to an entity.
#[derive(Debug, Clone)]
pub struct GpuParticleComponent {
    /// Name of the particle system (for lookup in the particle system manager).
    pub system_name: String,
    /// Local-space offset from the entity's transform.
    pub offset: Vec3,
    /// Whether this component is currently active.
    pub active: bool,
}

impl GpuParticleComponent {
    /// Creates a new component referencing a named particle system.
    pub fn new(system_name: impl Into<String>) -> Self {
        Self {
            system_name: system_name.into(),
            offset: Vec3::ZERO,
            active: true,
        }
    }

    /// Sets the local-space offset.
    pub fn with_offset(mut self, offset: Vec3) -> Self {
        self.offset = offset;
        self
    }
}

// ---------------------------------------------------------------------------
// GpuParticleManager
// ---------------------------------------------------------------------------

/// Manages multiple named GPU particle systems.
///
/// Typically there is one manager per scene. Systems are registered by name
/// and can be looked up, updated, and rendered collectively.
#[derive(Debug, Default)]
pub struct GpuParticleManager {
    systems: HashMap<String, GpuParticleSystem>,
}

impl GpuParticleManager {
    /// Creates a new empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a particle system. Returns the old system if one with the
    /// same name already existed.
    pub fn register(
        &mut self,
        system: GpuParticleSystem,
    ) -> Option<GpuParticleSystem> {
        self.systems.insert(system.name.clone(), system)
    }

    /// Removes a particle system by name.
    pub fn unregister(&mut self, name: &str) -> Option<GpuParticleSystem> {
        self.systems.remove(name)
    }

    /// Returns a reference to a system by name.
    pub fn get(&self, name: &str) -> Option<&GpuParticleSystem> {
        self.systems.get(name)
    }

    /// Returns a mutable reference to a system by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut GpuParticleSystem> {
        self.systems.get_mut(name)
    }

    /// Returns an iterator over all systems.
    pub fn systems(&self) -> impl Iterator<Item = (&str, &GpuParticleSystem)> {
        self.systems.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Prepares frame data for all systems.
    pub fn prepare_all_frames(
        &mut self,
        dt: f32,
        view_matrix: Mat4,
    ) -> Vec<(String, GpuParticleFrameData)> {
        self.systems
            .iter_mut()
            .map(|(name, system)| {
                let data = system.prepare_frame(dt, view_matrix);
                (name.clone(), data)
            })
            .collect()
    }

    /// Pauses all systems.
    pub fn pause_all(&mut self) {
        for system in self.systems.values_mut() {
            system.pause();
        }
    }

    /// Resumes all systems.
    pub fn resume_all(&mut self) {
        for system in self.systems.values_mut() {
            system.resume();
        }
    }

    /// Resets all systems.
    pub fn reset_all(&mut self) {
        for system in self.systems.values_mut() {
            system.reset();
        }
    }

    /// Returns the total GPU memory required across all systems.
    pub fn total_gpu_memory(&self) -> u64 {
        self.systems
            .values()
            .map(|s| s.buffer_sizes().total())
            .sum()
    }

    /// Returns the number of registered systems.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = GpuParticleSettings::default();
        assert_eq!(settings.max_particles, 65536);
        assert!(settings.depth_sort);
    }

    #[test]
    fn test_emitter_shape_type_index() {
        assert_eq!(EmitterShape::Point.type_index(), 0);
        assert_eq!(EmitterShape::Sphere { radius: 1.0 }.type_index(), 1);
        assert_eq!(
            EmitterShape::Box {
                half_extents: Vec3::ONE
            }
            .type_index(),
            2
        );
        assert_eq!(EmitterShape::Cone { radius: 1.0 }.type_index(), 3);
    }

    #[test]
    fn test_emitter_prepare_frame() {
        let settings = GpuParticleSettings {
            emit_rate: 100.0,
            ..Default::default()
        };
        let mut emitter = GpuEmitter::new(Vec3::ZERO, settings);
        let data = emitter.prepare_frame(1.0 / 60.0);
        // At 100 particles/sec, ~1.67 particles per frame at 60fps
        assert!(data.emit_count <= 2);
    }

    #[test]
    fn test_counter_data_initial() {
        let counter = CounterData::new_initial(1024);
        assert_eq!(counter.alive_count, 0);
        assert_eq!(counter.dead_count, 1024);
        assert_eq!(counter.draw_vertex_count, 6);
    }

    #[test]
    fn test_bitonic_sort_passes() {
        let total = BitonicSortPass::total_passes(8);
        assert_eq!(total, 6); // log2(8) = 3, 3*(3+1)/2 = 6

        let mut pass = BitonicSortPass::new(8);
        let mut count = 0;
        while pass.next_pass().is_some() {
            count += 1;
        }
        assert_eq!(count, 6);
    }

    #[test]
    fn test_bitonic_sort_single_element() {
        let total = BitonicSortPass::total_passes(1);
        assert_eq!(total, 0);
    }

    #[test]
    fn test_buffer_sizes() {
        let system = GpuParticleSystem::new("test", 1024);
        let sizes = system.buffer_sizes();
        assert_eq!(sizes.particle_buffer, 1024 * 64);
        assert_eq!(sizes.alive_index_buffer, 1024 * 4);
        assert_eq!(sizes.counter_buffer, 32);
        assert!(sizes.total() > 0);
    }

    #[test]
    fn test_system_lifecycle() {
        let mut system = GpuParticleSystem::new("fire", 10000);
        let emitter = GpuEmitter::new(Vec3::ZERO, GpuParticleSettings::fire());
        system.add_emitter("main", emitter);
        assert!(system.emitter("main").is_some());
        assert!(system.emitter("nonexistent").is_none());

        system.pause();
        assert!(system.is_paused());
        system.resume();
        assert!(!system.is_paused());

        system.remove_emitter("main");
        assert!(system.emitter("main").is_none());
    }

    #[test]
    fn test_manager() {
        let mut mgr = GpuParticleManager::new();
        let sys = GpuParticleSystem::new("fire", 1000);
        mgr.register(sys);
        assert_eq!(mgr.system_count(), 1);
        assert!(mgr.get("fire").is_some());

        mgr.pause_all();
        assert!(mgr.get("fire").unwrap().is_paused());

        mgr.unregister("fire");
        assert_eq!(mgr.system_count(), 0);
    }

    #[test]
    fn test_preset_settings() {
        let fire = GpuParticleSettings::fire();
        assert!(fire.noise_strength > 0.0);

        let rain = GpuParticleSettings::rain();
        assert!(!rain.depth_sort);

        let snow = GpuParticleSettings::snow();
        assert!(snow.min_lifetime > rain.min_lifetime);

        let sparkle = GpuParticleSettings::sparkle();
        assert!(sparkle.spread_angle > fire.spread_angle);
    }

    #[test]
    fn test_workgroup_counts() {
        let system = GpuParticleSystem::new("test", 1000);
        assert_eq!(system.update_workgroups(), 4); // ceil(1000/256)
        assert_eq!(GpuParticleSystem::emit_workgroups(100), 1);
        assert_eq!(GpuParticleSystem::emit_workgroups(257), 2);
    }

    #[test]
    fn test_component() {
        let comp = GpuParticleComponent::new("fire").with_offset(Vec3::Y);
        assert_eq!(comp.system_name, "fire");
        assert_eq!(comp.offset, Vec3::Y);
        assert!(comp.active);
    }

    #[test]
    fn test_initial_dead_indices() {
        let system = GpuParticleSystem::new("test", 16);
        let indices = system.initial_dead_indices();
        assert_eq!(indices.len(), 16);
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(idx, i as u32);
        }
    }
}
