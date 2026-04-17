//! # Audio Occlusion and Obstruction
//!
//! Simulates how sound propagates through a game world, accounting for walls,
//! obstacles, portals, and material properties.
//!
//! ## Features
//!
//! - **Ray-based obstruction** — Cast rays between listener and source to detect
//!   direct path blockage and apply volume/filter attenuation.
//! - **Diffraction** — Sound bends around corners via edge detection. When the
//!   direct path is blocked, the system finds diffraction edges and computes
//!   an alternate path.
//! - **Transmission** — Sound passes through walls with material-based attenuation.
//!   Different materials (glass, concrete, wood) absorb different frequencies.
//! - **Portal acoustics** — Room-based propagation through portal openings
//!   (doors, windows) with per-portal gain and filtering.
//! - **Low-pass filter** — Occluded sounds are attenuated primarily in high
//!   frequencies, simulating real-world muffling.

use std::collections::HashMap;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Vec3 (local minimal math type)
// ---------------------------------------------------------------------------

/// Minimal 3D vector for audio occlusion calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// Create a new vector.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Zero vector.
    pub const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);

    /// Length (magnitude) of the vector.
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Squared length.
    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Normalize the vector.
    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-8 {
            Self::ZERO
        } else {
            Self {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        }
    }

    /// Dot product.
    pub fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Distance to another point.
    pub fn distance(&self, other: &Vec3) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Lerp between two vectors.
    pub fn lerp(a: &Vec3, b: &Vec3, t: f32) -> Vec3 {
        Vec3 {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            z: a.z + (b.z - a.z) * t,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

// ---------------------------------------------------------------------------
// AcousticMaterial
// ---------------------------------------------------------------------------

/// Material properties that affect sound transmission.
#[derive(Debug, Clone)]
pub struct AcousticMaterial {
    /// Material name.
    pub name: String,
    /// Transmission loss in dB per unit thickness at low frequencies (< 500 Hz).
    pub low_freq_loss_db: f32,
    /// Transmission loss in dB per unit thickness at mid frequencies (500 Hz - 2 kHz).
    pub mid_freq_loss_db: f32,
    /// Transmission loss in dB per unit thickness at high frequencies (> 2 kHz).
    pub high_freq_loss_db: f32,
    /// Reflection coefficient [0, 1]. How much sound bounces off the surface.
    pub reflection: f32,
    /// Absorption coefficient [0, 1]. How much sound energy is absorbed.
    pub absorption: f32,
    /// Diffusion coefficient [0, 1]. How much reflected sound is scattered.
    pub diffusion: f32,
}

impl AcousticMaterial {
    /// Create a new acoustic material.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            low_freq_loss_db: 10.0,
            mid_freq_loss_db: 20.0,
            high_freq_loss_db: 30.0,
            reflection: 0.5,
            absorption: 0.3,
            diffusion: 0.2,
        }
    }

    /// Concrete wall — heavy, high attenuation.
    pub fn concrete() -> Self {
        Self {
            name: "Concrete".to_string(),
            low_freq_loss_db: 30.0,
            mid_freq_loss_db: 45.0,
            high_freq_loss_db: 55.0,
            reflection: 0.8,
            absorption: 0.05,
            diffusion: 0.1,
        }
    }

    /// Glass — moderate attenuation, high reflection.
    pub fn glass() -> Self {
        Self {
            name: "Glass".to_string(),
            low_freq_loss_db: 15.0,
            mid_freq_loss_db: 25.0,
            high_freq_loss_db: 30.0,
            reflection: 0.9,
            absorption: 0.02,
            diffusion: 0.05,
        }
    }

    /// Wood — light attenuation.
    pub fn wood() -> Self {
        Self {
            name: "Wood".to_string(),
            low_freq_loss_db: 10.0,
            mid_freq_loss_db: 18.0,
            high_freq_loss_db: 25.0,
            reflection: 0.6,
            absorption: 0.15,
            diffusion: 0.3,
        }
    }

    /// Drywall — light residential wall.
    pub fn drywall() -> Self {
        Self {
            name: "Drywall".to_string(),
            low_freq_loss_db: 8.0,
            mid_freq_loss_db: 15.0,
            high_freq_loss_db: 20.0,
            reflection: 0.5,
            absorption: 0.1,
            diffusion: 0.2,
        }
    }

    /// Metal — very high reflection, moderate transmission loss.
    pub fn metal() -> Self {
        Self {
            name: "Metal".to_string(),
            low_freq_loss_db: 25.0,
            mid_freq_loss_db: 40.0,
            high_freq_loss_db: 50.0,
            reflection: 0.95,
            absorption: 0.01,
            diffusion: 0.05,
        }
    }

    /// Fabric / curtain — low reflection, moderate absorption.
    pub fn fabric() -> Self {
        Self {
            name: "Fabric".to_string(),
            low_freq_loss_db: 3.0,
            mid_freq_loss_db: 8.0,
            high_freq_loss_db: 12.0,
            reflection: 0.2,
            absorption: 0.5,
            diffusion: 0.7,
        }
    }

    /// Compute the total transmission loss for a wall of given thickness.
    pub fn transmission_loss(&self, thickness: f32) -> TransmissionLoss {
        TransmissionLoss {
            low_freq_db: self.low_freq_loss_db * thickness,
            mid_freq_db: self.mid_freq_loss_db * thickness,
            high_freq_db: self.high_freq_loss_db * thickness,
        }
    }
}

/// Frequency-dependent transmission loss.
#[derive(Debug, Clone, Copy)]
pub struct TransmissionLoss {
    /// Loss at low frequencies in dB.
    pub low_freq_db: f32,
    /// Loss at mid frequencies in dB.
    pub mid_freq_db: f32,
    /// Loss at high frequencies in dB.
    pub high_freq_db: f32,
}

impl TransmissionLoss {
    /// Compute an average loss across all bands.
    pub fn average_db(&self) -> f32 {
        (self.low_freq_db + self.mid_freq_db + self.high_freq_db) / 3.0
    }

    /// Convert dB loss to a linear gain [0, 1].
    pub fn to_linear_gain(&self) -> f32 {
        let avg_db = self.average_db();
        10.0_f32.powf(-avg_db / 20.0)
    }

    /// Compute low-pass filter cutoff frequency based on frequency-dependent loss.
    ///
    /// Higher high-frequency loss relative to low-frequency loss means a lower
    /// cutoff (more muffled sound).
    pub fn to_lpf_cutoff(&self) -> f32 {
        let ratio = self.high_freq_db / self.low_freq_db.max(0.1);
        // Map ratio to cutoff: higher ratio = lower cutoff.
        let cutoff = 20000.0 / (1.0 + ratio * 0.5);
        cutoff.clamp(200.0, 20000.0)
    }
}

// ---------------------------------------------------------------------------
// OcclusionWall
// ---------------------------------------------------------------------------

/// An occluding wall/surface in the world.
#[derive(Debug, Clone)]
pub struct OcclusionWall {
    /// Unique identifier.
    pub id: u32,
    /// Position of the wall center.
    pub position: Vec3,
    /// Normal of the wall surface (facing the "front" side).
    pub normal: Vec3,
    /// Half-extents of the wall.
    pub half_extents: Vec3,
    /// Wall thickness.
    pub thickness: f32,
    /// Material of the wall.
    pub material: AcousticMaterial,
    /// Whether this wall is currently enabled.
    pub enabled: bool,
}

impl OcclusionWall {
    /// Create a new occlusion wall.
    pub fn new(
        id: u32,
        position: Vec3,
        normal: Vec3,
        half_extents: Vec3,
        thickness: f32,
        material: AcousticMaterial,
    ) -> Self {
        Self {
            id,
            position,
            normal: normal.normalized(),
            half_extents,
            thickness,
            material,
            enabled: true,
        }
    }

    /// Test if a ray from `origin` in direction `dir` hits this wall.
    ///
    /// Returns the distance to the intersection point, or None if missed.
    pub fn ray_intersect(&self, origin: &Vec3, dir: &Vec3) -> Option<f32> {
        // Plane intersection test.
        let denom = self.normal.dot(dir);
        if denom.abs() < 1e-6 {
            return None; // Parallel.
        }

        let diff = self.position - *origin;
        let t = diff.dot(&self.normal) / denom;

        if t < 0.0 {
            return None; // Behind the ray.
        }

        // Check if the hit point is within the wall's bounds.
        let hit = *origin + *dir * t;
        let local = hit - self.position;

        // Simple AABB check in wall-local space.
        if local.x.abs() <= self.half_extents.x
            && local.y.abs() <= self.half_extents.y
            && local.z.abs() <= self.half_extents.z
        {
            Some(t)
        } else {
            None
        }
    }

    /// Compute the transmission loss through this wall.
    pub fn compute_transmission_loss(&self) -> TransmissionLoss {
        self.material.transmission_loss(self.thickness)
    }
}

// ---------------------------------------------------------------------------
// DiffractionEdge
// ---------------------------------------------------------------------------

/// An edge where sound can diffract (bend around a corner).
#[derive(Debug, Clone)]
pub struct DiffractionEdge {
    /// Unique identifier.
    pub id: u32,
    /// Start point of the edge.
    pub start: Vec3,
    /// End point of the edge.
    pub end: Vec3,
    /// The direction perpendicular to the edge (outward from the occluder).
    pub outward_normal: Vec3,
    /// Diffraction loss factor [0, 1]. 0 = no diffraction, 1 = full pass-through.
    pub diffraction_factor: f32,
}

impl DiffractionEdge {
    /// Create a new diffraction edge.
    pub fn new(id: u32, start: Vec3, end: Vec3, outward_normal: Vec3) -> Self {
        Self {
            id,
            start,
            end,
            outward_normal: outward_normal.normalized(),
            diffraction_factor: 0.3,
        }
    }

    /// Find the closest point on this edge to a given point.
    pub fn closest_point(&self, point: &Vec3) -> Vec3 {
        let edge = self.end - self.start;
        let len_sq = edge.length_sq();
        if len_sq < 1e-8 {
            return self.start;
        }

        let t = (*point - self.start).dot(&edge) / len_sq;
        let t = t.clamp(0.0, 1.0);

        self.start + edge * t
    }

    /// Compute the diffraction path length: source -> edge -> listener.
    pub fn path_length(&self, source: &Vec3, listener: &Vec3) -> f32 {
        let edge_point = self.closest_point(source);
        source.distance(&edge_point) + edge_point.distance(listener)
    }

    /// Compute the diffraction angle (in radians).
    ///
    /// This is the angle between the source-edge and edge-listener directions,
    /// projected onto the plane perpendicular to the edge.
    pub fn diffraction_angle(&self, source: &Vec3, listener: &Vec3) -> f32 {
        let edge_point = self.closest_point(source);
        let to_source = (*source - edge_point).normalized();
        let to_listener = (*listener - edge_point).normalized();

        let cos_angle = to_source.dot(&to_listener).clamp(-1.0, 1.0);
        cos_angle.acos()
    }

    /// Compute the diffraction attenuation based on angle.
    ///
    /// Uses a simplified UTD (Uniform Theory of Diffraction) model.
    pub fn compute_attenuation(&self, source: &Vec3, listener: &Vec3) -> f32 {
        let angle = self.diffraction_angle(source, listener);
        let normalized_angle = angle / PI;

        // Higher angle = more attenuation.
        let attenuation = (1.0 - normalized_angle * 0.7).clamp(0.0, 1.0);
        attenuation * self.diffraction_factor
    }
}

// ---------------------------------------------------------------------------
// Portal
// ---------------------------------------------------------------------------

/// A portal (opening) between two acoustic rooms.
///
/// Sound propagates through portals with gain and optional filtering.
/// Doors, windows, and hallway openings are typical portals.
#[derive(Debug, Clone)]
pub struct AcousticPortal {
    /// Unique identifier.
    pub id: u32,
    /// Name for debugging.
    pub name: String,
    /// Center position of the portal opening.
    pub position: Vec3,
    /// Normal of the portal (pointing into the destination room).
    pub normal: Vec3,
    /// Width of the opening.
    pub width: f32,
    /// Height of the opening.
    pub height: f32,
    /// Room ID on side A.
    pub room_a: u32,
    /// Room ID on side B.
    pub room_b: u32,
    /// Gain multiplier [0, 1]. 0 = fully closed, 1 = fully open.
    pub openness: f32,
    /// Low-pass filter cutoff when partially closed.
    pub lpf_cutoff: f32,
    /// Whether the portal is active.
    pub enabled: bool,
}

impl AcousticPortal {
    /// Create a new portal.
    pub fn new(
        id: u32,
        name: impl Into<String>,
        position: Vec3,
        normal: Vec3,
        width: f32,
        height: f32,
        room_a: u32,
        room_b: u32,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            position,
            normal: normal.normalized(),
            width,
            height,
            room_a,
            room_b,
            openness: 1.0,
            lpf_cutoff: 20000.0,
            enabled: true,
        }
    }

    /// Set the openness (0 = closed, 1 = fully open).
    pub fn set_openness(&mut self, openness: f32) {
        self.openness = openness.clamp(0.0, 1.0);
        // Auto-adjust LPF based on openness.
        self.lpf_cutoff = 200.0 + (20000.0 - 200.0) * self.openness;
    }

    /// Returns the area of the opening.
    pub fn area(&self) -> f32 {
        self.width * self.height * self.openness
    }

    /// Returns the gain contributed by this portal.
    pub fn gain(&self) -> f32 {
        self.openness
    }

    /// Returns the room on the other side of the portal.
    pub fn other_room(&self, current_room: u32) -> Option<u32> {
        if current_room == self.room_a {
            Some(self.room_b)
        } else if current_room == self.room_b {
            Some(self.room_a)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// AcousticRoom
// ---------------------------------------------------------------------------

/// A room in the acoustic simulation.
///
/// Rooms define enclosed volumes where sound behaves consistently.
/// Portals connect rooms for inter-room propagation.
#[derive(Debug, Clone)]
pub struct AcousticRoom {
    /// Unique room identifier.
    pub id: u32,
    /// Room name.
    pub name: String,
    /// Center position.
    pub center: Vec3,
    /// Half-extents (AABB approximation).
    pub half_extents: Vec3,
    /// Connected portal IDs.
    pub portals: Vec<u32>,
    /// Reverb time (RT60) for this room.
    pub reverb_time: f32,
    /// Ambient sound level in dB.
    pub ambient_level_db: f32,
}

impl AcousticRoom {
    /// Create a new room.
    pub fn new(id: u32, name: impl Into<String>, center: Vec3, half_extents: Vec3) -> Self {
        Self {
            id,
            name: name.into(),
            center,
            half_extents,
            portals: Vec::new(),
            reverb_time: 0.5,
            ambient_level_db: -40.0,
        }
    }

    /// Check if a point is inside this room.
    pub fn contains(&self, point: &Vec3) -> bool {
        let local = *point - self.center;
        local.x.abs() <= self.half_extents.x
            && local.y.abs() <= self.half_extents.y
            && local.z.abs() <= self.half_extents.z
    }

    /// Approximate volume of the room.
    pub fn volume(&self) -> f32 {
        8.0 * self.half_extents.x * self.half_extents.y * self.half_extents.z
    }

    /// Add a portal to this room.
    pub fn add_portal(&mut self, portal_id: u32) {
        if !self.portals.contains(&portal_id) {
            self.portals.push(portal_id);
        }
    }
}

// ---------------------------------------------------------------------------
// OcclusionResult
// ---------------------------------------------------------------------------

/// Result of an occlusion query for a single sound source.
#[derive(Debug, Clone)]
pub struct OcclusionResult {
    /// Volume gain [0, 1] after obstruction.
    pub gain: f32,
    /// Low-pass filter cutoff frequency (Hz).
    pub lpf_cutoff: f32,
    /// Whether the direct path is fully blocked.
    pub is_fully_occluded: bool,
    /// Whether sound arrives via diffraction.
    pub has_diffraction: bool,
    /// Whether sound arrives via portal propagation.
    pub has_portal_path: bool,
    /// Whether sound arrives via transmission through walls.
    pub has_transmission: bool,
    /// Number of walls the sound passed through.
    pub walls_traversed: u32,
    /// Total path length (direct or diffracted).
    pub path_length: f32,
    /// Number of portals in the path.
    pub portal_count: u32,
}

impl OcclusionResult {
    /// Create a fully unoccluded result (clear line of sight).
    pub fn clear(distance: f32) -> Self {
        Self {
            gain: 1.0,
            lpf_cutoff: 20000.0,
            is_fully_occluded: false,
            has_diffraction: false,
            has_portal_path: false,
            has_transmission: false,
            walls_traversed: 0,
            path_length: distance,
            portal_count: 0,
        }
    }

    /// Create a fully occluded result (completely blocked).
    pub fn blocked() -> Self {
        Self {
            gain: 0.0,
            lpf_cutoff: 200.0,
            is_fully_occluded: true,
            has_diffraction: false,
            has_portal_path: false,
            has_transmission: false,
            walls_traversed: 0,
            path_length: f32::INFINITY,
            portal_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// OcclusionConfig
// ---------------------------------------------------------------------------

/// Configuration for the occlusion system.
#[derive(Debug, Clone)]
pub struct OcclusionConfig {
    /// Maximum number of walls to consider for transmission.
    pub max_wall_transmission: u32,
    /// Maximum number of diffraction edges to evaluate.
    pub max_diffraction_edges: u32,
    /// Maximum portal propagation depth (rooms to traverse).
    pub max_portal_depth: u32,
    /// Minimum gain below which sound is considered inaudible.
    pub min_audible_gain: f32,
    /// Whether to enable diffraction.
    pub enable_diffraction: bool,
    /// Whether to enable transmission.
    pub enable_transmission: bool,
    /// Whether to enable portal propagation.
    pub enable_portals: bool,
    /// Low-pass filter minimum cutoff (Hz).
    pub min_lpf_cutoff: f32,
    /// Diffraction attenuation multiplier.
    pub diffraction_strength: f32,
    /// Transmission attenuation multiplier.
    pub transmission_strength: f32,
}

impl Default for OcclusionConfig {
    fn default() -> Self {
        Self {
            max_wall_transmission: 3,
            max_diffraction_edges: 4,
            max_portal_depth: 3,
            min_audible_gain: 0.001,
            enable_diffraction: true,
            enable_transmission: true,
            enable_portals: true,
            min_lpf_cutoff: 200.0,
            diffraction_strength: 1.0,
            transmission_strength: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// LowPassFilter
// ---------------------------------------------------------------------------

/// Simple one-pole low-pass filter for occluded sounds.
///
/// When sound is occluded, high frequencies are attenuated more than low
/// frequencies, creating a muffled effect.
#[derive(Debug, Clone)]
pub struct OcclusionLowPassFilter {
    /// Cutoff frequency in Hz.
    pub cutoff: f32,
    /// Sample rate.
    pub sample_rate: u32,
    /// Filter coefficient.
    alpha: f32,
    /// Previous output sample (per channel).
    prev_samples: Vec<f32>,
    /// Number of channels.
    channels: usize,
}

impl OcclusionLowPassFilter {
    /// Create a new low-pass filter.
    pub fn new(cutoff: f32, sample_rate: u32, channels: usize) -> Self {
        let alpha = Self::compute_alpha(cutoff, sample_rate);
        Self {
            cutoff,
            sample_rate,
            alpha,
            prev_samples: vec![0.0; channels],
            channels,
        }
    }

    /// Compute the filter coefficient from cutoff and sample rate.
    fn compute_alpha(cutoff: f32, sample_rate: u32) -> f32 {
        let rc = 1.0 / (2.0 * PI * cutoff);
        let dt = 1.0 / sample_rate as f32;
        dt / (rc + dt)
    }

    /// Set the cutoff frequency.
    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff.clamp(20.0, 20000.0);
        self.alpha = Self::compute_alpha(self.cutoff, self.sample_rate);
    }

    /// Process interleaved audio samples in-place.
    pub fn process(&mut self, samples: &mut [f32]) {
        let channels = self.channels;
        if channels == 0 {
            return;
        }

        for (i, sample) in samples.iter_mut().enumerate() {
            let ch = i % channels;
            let output = self.prev_samples[ch] + self.alpha * (*sample - self.prev_samples[ch]);
            self.prev_samples[ch] = output;
            *sample = output;
        }
    }

    /// Process mono samples in-place.
    pub fn process_mono(&mut self, samples: &mut [f32]) {
        let mut prev = self.prev_samples[0];
        for sample in samples.iter_mut() {
            let output = prev + self.alpha * (*sample - prev);
            prev = output;
            *sample = output;
        }
        self.prev_samples[0] = prev;
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        for s in &mut self.prev_samples {
            *s = 0.0;
        }
    }

    /// Returns the current cutoff.
    pub fn cutoff(&self) -> f32 {
        self.cutoff
    }
}

// ---------------------------------------------------------------------------
// OcclusionSystem
// ---------------------------------------------------------------------------

/// The main audio occlusion system.
///
/// Manages walls, edges, portals, and rooms. Queries occlusion between
/// any source-listener pair.
pub struct OcclusionSystem {
    /// Configuration.
    config: OcclusionConfig,
    /// All occlusion walls.
    walls: Vec<OcclusionWall>,
    /// Diffraction edges.
    edges: Vec<DiffractionEdge>,
    /// Portals.
    portals: Vec<AcousticPortal>,
    /// Rooms.
    rooms: Vec<AcousticRoom>,
    /// Material library.
    materials: HashMap<String, AcousticMaterial>,
    /// Per-source low-pass filters.
    filters: HashMap<u32, OcclusionLowPassFilter>,
    /// Cache of recent occlusion results.
    cache: HashMap<(u32, u32), OcclusionResult>,
    /// Whether the cache is dirty (geometry changed).
    cache_dirty: bool,
}

impl OcclusionSystem {
    /// Create a new occlusion system.
    pub fn new(config: OcclusionConfig) -> Self {
        Self {
            config,
            walls: Vec::new(),
            edges: Vec::new(),
            portals: Vec::new(),
            rooms: Vec::new(),
            materials: HashMap::new(),
            filters: HashMap::new(),
            cache: HashMap::new(),
            cache_dirty: true,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(OcclusionConfig::default())
    }

    // -------------------------------------------------------------------
    // Geometry management
    // -------------------------------------------------------------------

    /// Add an occlusion wall.
    pub fn add_wall(&mut self, wall: OcclusionWall) {
        self.walls.push(wall);
        self.cache_dirty = true;
    }

    /// Remove a wall by ID.
    pub fn remove_wall(&mut self, id: u32) -> bool {
        let before = self.walls.len();
        self.walls.retain(|w| w.id != id);
        let removed = self.walls.len() < before;
        if removed {
            self.cache_dirty = true;
        }
        removed
    }

    /// Add a diffraction edge.
    pub fn add_edge(&mut self, edge: DiffractionEdge) {
        self.edges.push(edge);
        self.cache_dirty = true;
    }

    /// Remove an edge by ID.
    pub fn remove_edge(&mut self, id: u32) -> bool {
        let before = self.edges.len();
        self.edges.retain(|e| e.id != id);
        let removed = self.edges.len() < before;
        if removed {
            self.cache_dirty = true;
        }
        removed
    }

    /// Add a portal.
    pub fn add_portal(&mut self, portal: AcousticPortal) {
        // Register with rooms.
        let room_a = portal.room_a;
        let room_b = portal.room_b;
        let portal_id = portal.id;

        self.portals.push(portal);

        for room in &mut self.rooms {
            if room.id == room_a || room.id == room_b {
                room.add_portal(portal_id);
            }
        }

        self.cache_dirty = true;
    }

    /// Add a room.
    pub fn add_room(&mut self, room: AcousticRoom) {
        self.rooms.push(room);
        self.cache_dirty = true;
    }

    /// Register an acoustic material.
    pub fn register_material(&mut self, material: AcousticMaterial) {
        self.materials.insert(material.name.clone(), material);
    }

    // -------------------------------------------------------------------
    // Query
    // -------------------------------------------------------------------

    /// Query the occlusion between a source and listener.
    ///
    /// Returns an `OcclusionResult` describing the gain, filter cutoff,
    /// and path information.
    pub fn query_occlusion(
        &self,
        source_pos: &Vec3,
        listener_pos: &Vec3,
    ) -> OcclusionResult {
        let direct_distance = source_pos.distance(listener_pos);
        if direct_distance < 0.01 {
            return OcclusionResult::clear(direct_distance);
        }

        let direction = (*listener_pos - *source_pos).normalized();

        // Cast ray from source to listener, find all wall intersections.
        let mut hit_walls: Vec<(&OcclusionWall, f32)> = Vec::new();
        for wall in &self.walls {
            if !wall.enabled {
                continue;
            }
            if let Some(t) = wall.ray_intersect(source_pos, &direction) {
                if t < direct_distance {
                    hit_walls.push((wall, t));
                }
            }
        }

        // Sort by distance.
        hit_walls.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // If no walls are hit, the path is clear.
        if hit_walls.is_empty() {
            return OcclusionResult::clear(direct_distance);
        }

        // Compute direct path gain from wall transmission.
        let mut result = self.compute_transmission(source_pos, listener_pos, &hit_walls);

        // Try diffraction if enabled and direct path is significantly attenuated.
        if self.config.enable_diffraction && result.gain < 0.5 {
            let diffraction_result = self.compute_diffraction(source_pos, listener_pos);
            if diffraction_result.gain > result.gain {
                result = diffraction_result;
            }
        }

        // Try portal propagation if enabled.
        if self.config.enable_portals {
            let portal_result = self.compute_portal_path(source_pos, listener_pos);
            if let Some(pr) = portal_result {
                if pr.gain > result.gain {
                    result = pr;
                }
            }
        }

        result
    }

    /// Compute transmission through walls.
    fn compute_transmission(
        &self,
        source_pos: &Vec3,
        listener_pos: &Vec3,
        hit_walls: &[(&OcclusionWall, f32)],
    ) -> OcclusionResult {
        let max_walls = self.config.max_wall_transmission as usize;
        let walls_to_process = hit_walls.len().min(max_walls);

        let mut total_gain = 1.0_f32;
        let mut total_lpf = 20000.0_f32;

        for (wall, _) in hit_walls.iter().take(walls_to_process) {
            let loss = wall.compute_transmission_loss();
            total_gain *= loss.to_linear_gain() * self.config.transmission_strength;
            total_lpf = total_lpf.min(loss.to_lpf_cutoff());
        }

        total_gain = total_gain.max(self.config.min_audible_gain);
        total_lpf = total_lpf.max(self.config.min_lpf_cutoff);

        OcclusionResult {
            gain: total_gain,
            lpf_cutoff: total_lpf,
            is_fully_occluded: total_gain <= self.config.min_audible_gain,
            has_diffraction: false,
            has_portal_path: false,
            has_transmission: true,
            walls_traversed: walls_to_process as u32,
            path_length: source_pos.distance(listener_pos),
            portal_count: 0,
        }
    }

    /// Compute diffraction around edges.
    fn compute_diffraction(
        &self,
        source_pos: &Vec3,
        listener_pos: &Vec3,
    ) -> OcclusionResult {
        let max_edges = self.config.max_diffraction_edges as usize;

        let mut best_gain = 0.0_f32;
        let mut best_path = f32::INFINITY;
        let mut best_cutoff = self.config.min_lpf_cutoff;

        for edge in self.edges.iter().take(max_edges) {
            let attenuation = edge.compute_attenuation(source_pos, listener_pos);
            let path_len = edge.path_length(source_pos, listener_pos);

            let gain = attenuation * self.config.diffraction_strength;
            if gain > best_gain {
                best_gain = gain;
                best_path = path_len;
                // Higher diffraction angle = more LPF.
                let angle = edge.diffraction_angle(source_pos, listener_pos);
                best_cutoff = 20000.0 * (1.0 - angle / PI * 0.8);
                best_cutoff = best_cutoff.clamp(self.config.min_lpf_cutoff, 20000.0);
            }
        }

        OcclusionResult {
            gain: best_gain,
            lpf_cutoff: best_cutoff,
            is_fully_occluded: best_gain <= self.config.min_audible_gain,
            has_diffraction: best_gain > 0.0,
            has_portal_path: false,
            has_transmission: false,
            walls_traversed: 0,
            path_length: best_path,
            portal_count: 0,
        }
    }

    /// Compute sound propagation through portals.
    fn compute_portal_path(
        &self,
        source_pos: &Vec3,
        listener_pos: &Vec3,
    ) -> Option<OcclusionResult> {
        // Find which rooms the source and listener are in.
        let source_room = self.rooms.iter().find(|r| r.contains(source_pos))?;
        let listener_room = self.rooms.iter().find(|r| r.contains(listener_pos))?;

        if source_room.id == listener_room.id {
            // Same room — portal propagation not needed.
            return None;
        }

        // BFS through portals to find a path from source room to listener room.
        let path = self.find_portal_path(
            source_room.id,
            listener_room.id,
            self.config.max_portal_depth,
        )?;

        // Compute gain along the portal chain.
        let mut total_gain = 1.0_f32;
        let mut min_cutoff = 20000.0_f32;
        let mut total_path = 0.0_f32;
        let mut current_pos = *source_pos;

        for &portal_id in &path {
            if let Some(portal) = self.portals.iter().find(|p| p.id == portal_id) {
                if !portal.enabled || portal.openness < 0.01 {
                    return None;
                }
                total_gain *= portal.gain();
                min_cutoff = min_cutoff.min(portal.lpf_cutoff);
                total_path += current_pos.distance(&portal.position);
                current_pos = portal.position;
            }
        }

        total_path += current_pos.distance(listener_pos);

        Some(OcclusionResult {
            gain: total_gain,
            lpf_cutoff: min_cutoff,
            is_fully_occluded: false,
            has_diffraction: false,
            has_portal_path: true,
            has_transmission: false,
            walls_traversed: 0,
            path_length: total_path,
            portal_count: path.len() as u32,
        })
    }

    /// BFS to find a portal path between two rooms.
    fn find_portal_path(
        &self,
        from_room: u32,
        to_room: u32,
        max_depth: u32,
    ) -> Option<Vec<u32>> {
        use std::collections::VecDeque;

        let mut visited = std::collections::HashSet::new();
        let mut queue: VecDeque<(u32, Vec<u32>)> = VecDeque::new();
        queue.push_back((from_room, Vec::new()));
        visited.insert(from_room);

        while let Some((current_room, path)) = queue.pop_front() {
            if path.len() >= max_depth as usize {
                continue;
            }

            // Find all portals connected to this room.
            for portal in &self.portals {
                if !portal.enabled {
                    continue;
                }
                if let Some(next_room) = portal.other_room(current_room) {
                    if next_room == to_room {
                        let mut full_path = path.clone();
                        full_path.push(portal.id);
                        return Some(full_path);
                    }
                    if !visited.contains(&next_room) {
                        visited.insert(next_room);
                        let mut new_path = path.clone();
                        new_path.push(portal.id);
                        queue.push_back((next_room, new_path));
                    }
                }
            }
        }

        None
    }

    // -------------------------------------------------------------------
    // Filter management
    // -------------------------------------------------------------------

    /// Get or create a low-pass filter for a sound source.
    pub fn get_filter(&mut self, source_id: u32, sample_rate: u32) -> &mut OcclusionLowPassFilter {
        self.filters
            .entry(source_id)
            .or_insert_with(|| OcclusionLowPassFilter::new(20000.0, sample_rate, 1))
    }

    /// Apply occlusion to an audio buffer.
    pub fn apply_occlusion(
        &mut self,
        source_id: u32,
        result: &OcclusionResult,
        samples: &mut [f32],
        sample_rate: u32,
    ) {
        // Apply gain.
        for sample in samples.iter_mut() {
            *sample *= result.gain;
        }

        // Apply low-pass filter.
        let filter = self.get_filter(source_id, sample_rate);
        filter.set_cutoff(result.lpf_cutoff);
        filter.process_mono(samples);
    }

    /// Remove a source's filter (when the source is destroyed).
    pub fn remove_filter(&mut self, source_id: u32) {
        self.filters.remove(&source_id);
    }

    // -------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------

    /// Returns the number of walls.
    pub fn wall_count(&self) -> usize {
        self.walls.len()
    }

    /// Returns the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns the number of portals.
    pub fn portal_count(&self) -> usize {
        self.portals.len()
    }

    /// Returns the number of rooms.
    pub fn room_count(&self) -> usize {
        self.rooms.len()
    }

    /// Invalidate the cache.
    pub fn invalidate_cache(&mut self) {
        self.cache.clear();
        self.cache_dirty = true;
    }

    /// Get the configuration.
    pub fn config(&self) -> &OcclusionConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: OcclusionConfig) {
        self.config = config;
        self.cache_dirty = true;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let sum = a + b;
        assert!((sum.x - 5.0).abs() < 1e-5);
        assert!((sum.y - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec3_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec3_normalized() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalized();
        assert!((n.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_material_presets() {
        let concrete = AcousticMaterial::concrete();
        assert!(concrete.low_freq_loss_db > 20.0);

        let glass = AcousticMaterial::glass();
        assert!(glass.reflection > 0.8);

        let fabric = AcousticMaterial::fabric();
        assert!(fabric.absorption > 0.3);
    }

    #[test]
    fn test_transmission_loss() {
        let material = AcousticMaterial::concrete();
        let loss = material.transmission_loss(0.2);
        assert!(loss.average_db() > 0.0);
        assert!(loss.to_linear_gain() < 1.0);
        assert!(loss.to_lpf_cutoff() < 20000.0);
    }

    #[test]
    fn test_wall_ray_intersect() {
        let wall = OcclusionWall::new(
            1,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(10.0, 10.0, 10.0),
            0.3,
            AcousticMaterial::concrete(),
        );

        let origin = Vec3::new(0.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let hit = wall.ray_intersect(&origin, &dir);
        assert!(hit.is_some());
    }

    #[test]
    fn test_diffraction_edge() {
        let edge = DiffractionEdge::new(
            1,
            Vec3::new(5.0, 0.0, -5.0),
            Vec3::new(5.0, 0.0, 5.0),
            Vec3::new(1.0, 0.0, 0.0),
        );

        let source = Vec3::new(0.0, 0.0, 0.0);
        let listener = Vec3::new(10.0, 0.0, 0.0);

        let atten = edge.compute_attenuation(&source, &listener);
        assert!(atten >= 0.0 && atten <= 1.0);
    }

    #[test]
    fn test_portal_other_room() {
        let portal = AcousticPortal::new(
            1,
            "door",
            Vec3::new(5.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            2.0,
            3.0,
            1,
            2,
        );

        assert_eq!(portal.other_room(1), Some(2));
        assert_eq!(portal.other_room(2), Some(1));
        assert_eq!(portal.other_room(3), None);
    }

    #[test]
    fn test_portal_openness() {
        let mut portal = AcousticPortal::new(
            1,
            "door",
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            2.0,
            3.0,
            1,
            2,
        );

        portal.set_openness(0.5);
        assert!((portal.openness - 0.5).abs() < 1e-5);
        assert!(portal.lpf_cutoff < 20000.0);
    }

    #[test]
    fn test_room_contains() {
        let room = AcousticRoom::new(
            1,
            "office",
            Vec3::new(5.0, 2.5, 5.0),
            Vec3::new(5.0, 2.5, 5.0),
        );

        assert!(room.contains(&Vec3::new(5.0, 2.5, 5.0)));
        assert!(!room.contains(&Vec3::new(20.0, 0.0, 0.0)));
    }

    #[test]
    fn test_occlusion_clear_path() {
        let system = OcclusionSystem::default_config();
        let source = Vec3::new(0.0, 0.0, 0.0);
        let listener = Vec3::new(10.0, 0.0, 0.0);

        let result = system.query_occlusion(&source, &listener);
        assert!((result.gain - 1.0).abs() < 1e-5);
        assert!(!result.is_fully_occluded);
    }

    #[test]
    fn test_occlusion_with_wall() {
        let mut system = OcclusionSystem::default_config();

        system.add_wall(OcclusionWall::new(
            1,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(10.0, 10.0, 10.0),
            0.3,
            AcousticMaterial::concrete(),
        ));

        let source = Vec3::new(0.0, 0.0, 0.0);
        let listener = Vec3::new(10.0, 0.0, 0.0);

        let result = system.query_occlusion(&source, &listener);
        assert!(result.gain < 1.0);
        assert!(result.has_transmission);
    }

    #[test]
    fn test_lpf_processing() {
        let mut filter = OcclusionLowPassFilter::new(1000.0, 44100, 1);

        // Process a step function — output should be smoothed.
        let mut samples = vec![1.0; 100];
        filter.process_mono(&mut samples);

        // The filter should ramp up rather than jump to 1.
        assert!(samples[0] < 1.0);
        // Later samples should be closer to 1.
        assert!(samples[99] > samples[0]);
    }

    #[test]
    fn test_occlusion_result_presets() {
        let clear = OcclusionResult::clear(10.0);
        assert!(!clear.is_fully_occluded);
        assert!((clear.gain - 1.0).abs() < 1e-5);

        let blocked = OcclusionResult::blocked();
        assert!(blocked.is_fully_occluded);
        assert!((blocked.gain - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_material_fabric() {
        let fabric = AcousticMaterial::fabric();
        let loss = fabric.transmission_loss(0.1);
        // Fabric is thin and absorbing, so loss should be low.
        assert!(loss.average_db() < 5.0);
    }
}
